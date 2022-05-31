import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from Attention import TempAttention, SpatialAttention
from memory_rand import MemoryRamTwoStreamModule2, MemoryRamModule2, MMModule2


class HME(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, max_len_v, max_len_q, device, input_drop_p=0.2):
        """
        Heterogeneous memory enhanced multimodal attention model for video question answering (CVPR19)
        """
        super(HME, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder


        dim = qns_encoder.dim_hidden

        self.temp_att_a = TempAttention(dim * 2, dim * 2, hidden_dim=256)
        self.temp_att_m = TempAttention(dim * 2, dim * 2, hidden_dim=256)
        self.mrm_vid = MemoryRamTwoStreamModule2(dim, dim, max_len_v, device)
        self.mrm_txt = MemoryRamModule2(dim, dim, max_len_q, device)

        self.mm_module_v1 = MMModule2(dim, input_drop_p, device)

        self.linear_vid = nn.Linear(dim*2, dim)
        self.linear_qns = nn.Linear(dim*2, dim)
        self.linear_mem = nn.Linear(dim*2, dim)
        self.vq2word_hme = nn.Linear(dim*3, 1)
        self.device = device

    def forward(self, video_appearance_feat, video_motion_feat, candidates, candidates_len, obj_feature, dep_adj, question, question_len, obj_fea_q, dep_adj_q):
        """
        Args:
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            candidates: [Tensor] (batch_size, 5, max_length, [emb_dim(for bert)])
            candidates_len: [Tensor] (batch_size, 5)
            obj_feature: [Tensor] (batch_size, 5, max_length, emb_dim)
            dep_adj: [Tensor] (batch_size, max_length, max_length)
            question: [Tensor] (batch_size, 5, max_length, [emb_dim(for bert)])
            question_len: [Tensor] (batch_size, 5)
            obj_fea_q: [Tensor] (batch_size, 5, max_length, emb_dim)
            dep_adj_q: [Tensor] (batch_size, max_length, max_length)
        return:
            logits, predict_idx
        """
        vid_feats = torch.cat([video_appearance_feat.mean(2), video_motion_feat], dim=-1)
        if self.qns_encoder.use_bert:
            candidates = candidates.permute(1, 0, 2, 3)  # for BERT
        else:
            candidates = candidates.permute(1, 0, 2)

        obj_feature = obj_feature.permute(1, 0, 2, 3)
        candidates_len = candidates_len.permute(1, 0)

        outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2 = self.vid_encoder(vid_feats)
        vid_feats = (outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2)

        qas_seq, qas_hidden = list(), list()
        for idx, qas in enumerate(candidates):
            q_output, s_hidden = self.qns_encoder(qas, candidates_len[idx], obj=obj_feature[idx])
            qas_seq.append(q_output)
            qas_hidden.append(s_hidden)

        out = []
        for idx, (qa_seq, qa_hidden) in enumerate(zip(qas_seq, qas_hidden)):
            encoder_out = self.vq_encoder(vid_feats, qa_seq, qa_hidden)
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)

        _, predict_idx = torch.max(out, 1)

        return out, predict_idx

    def vq_encoder(self, vid_feats, qns_seq, qns_hidden, iter_num=3):

        outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2 = vid_feats
        outputs_app = torch.cat((outputs_app_l1, outputs_app_l2), dim=-1)
        outputs_motion = torch.cat((outputs_motion_l1, outputs_motion_l2), dim=-1)

        batch_size, fnum, vid_feat_dim = outputs_app.size()

        batch_size, seq_len, qns_feat_dim = qns_seq.size()

        qns_embed = qns_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1) 

        # Apply temporal attention
        att_app, beta_app = self.temp_att_a(qns_embed, outputs_app)
        att_motion, beta_motion = self.temp_att_m(qns_embed, outputs_motion)
        tmp_app_motion = torch.cat((outputs_app_l2[:, -1, :], outputs_motion_l2[:, -1, :]), dim=-1)

        mem_output = torch.zeros(batch_size, vid_feat_dim).to(self.device)

        mem_ram_vid = self.mrm_vid(outputs_app_l2, outputs_motion_l2, fnum)
        mem_ram_txt = self.mrm_txt(qns_seq, qns_seq.shape[1])
        mem_output[:] = self.mm_module_v1(tmp_app_motion, mem_ram_vid, mem_ram_txt, iter_num)

        app_trans = torch.tanh(self.linear_vid(att_app))
        motion_trans = torch.tanh(self.linear_vid(att_motion))
        mem_trans = torch.tanh(self.linear_mem(mem_output))

        encoder_outputs = torch.cat((app_trans, motion_trans, mem_trans), dim=1)
        outputs = self.vq2word_hme(encoder_outputs).squeeze()

        return outputs