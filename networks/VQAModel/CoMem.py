import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from memory_module import EpisodicMemory


class CoMem(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, max_len_v, max_len_q, device):
        """
        motion-appearance co-memory networks for video question answering (CVPR18)
        """
        super(CoMem, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder

        dim = qns_encoder.dim_hidden

        self.epm_app = EpisodicMemory(dim*2)
        self.epm_mot = EpisodicMemory(dim*2)

        self.linear_ma = nn.Linear(dim*2*4, dim*2)
        self.linear_mb = nn.Linear(dim*2*4, dim*2)

        self.vq2word = nn.Linear(dim*2*2, 1)

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

        _, qns_hidden = self.qns_encoder(question, question_len, obj=obj_fea_q)
        qas_hidden = list()
        for idx, qas in enumerate(candidates):
            _, ah_tmp = self.qns_encoder(qas, candidates_len[idx], obj=obj_feature[idx])
            qas_hidden.append(ah_tmp)

        out = []
        for idx, qas in enumerate(qas_hidden):
            encoder_out = self.vq_encoder(vid_feats, qns_hidden, qas)
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)

        _, predict_idx = torch.max(out, 1)


        return out, predict_idx

    def vq_encoder(self, vid_feats, ques, qas, iter_num=3):

        outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2 = vid_feats

        outputs_app = torch.cat((outputs_app_l1, outputs_app_l2), dim=-1)
        outputs_motion = torch.cat((outputs_motion_l1, outputs_motion_l2), dim=-1)

        batch_size = qas.shape[1]

        qns_embed = ques.permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)
        qas_embed = qas.permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)

        m_app = outputs_app[:, -1, :]
        m_mot = outputs_motion[:, -1, :]
        ma, mb = m_app.detach(), m_mot.detach()
        m_app = m_app.unsqueeze(1)
        m_mot = m_mot.unsqueeze(1)
        for _ in range(iter_num):
            mm = ma + mb
            m_app = self.epm_app(outputs_app, mm, m_app)
            m_mot = self.epm_mot(outputs_motion, mm, m_mot)
            ma_q = torch.cat((ma, m_app.squeeze(1), qns_embed, qas_embed), dim=1)
            mb_q = torch.cat((mb, m_mot.squeeze(1), qns_embed, qas_embed), dim=1)
            ma = torch.tanh(self.linear_ma(ma_q))
            mb = torch.tanh(self.linear_mb(mb_q))

        mem = torch.cat((ma, mb), dim=1)
        outputs = self.vq2word(mem).squeeze()

        return outputs