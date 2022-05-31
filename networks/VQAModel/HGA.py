import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from networks.Transformer import CoAttention
from networks.GCN import AdjLearner, GCN
from block import fusions #pytorch >= 1.1.0


class HGA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device):
        """
        Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI2020)
        """
        super(HGA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        hidden_size = vid_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p

        self.co_attn = CoAttention(
            hidden_size, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)

        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=1,
            dropout=input_dropout_p)

        self.gcn_atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-1)) #change to dim=-2 for attention-pooling otherwise sum-pooling

        self.global_fusion = fusions.Block(
            [hidden_size, hidden_size], hidden_size, dropout_input=input_dropout_p)

        self.fusion = fusions.Block([hidden_size, hidden_size], 1)


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
        cand_len = candidates_len.permute(1, 0)

        v_output, v_hidden = self.vid_encoder(vid_feats)
        v_last_hidden = torch.squeeze(v_hidden)


        out = []
        for idx, qas in enumerate(candidates):
            encoder_out = self.vq_encoder(v_output, v_last_hidden, qas, cand_len[idx], obj_feature[idx])
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)
        _, predict_idx = torch.max(out, 1)

        return out, predict_idx


    def vq_encoder(self, v_output, v_last_hidden, qas, qas_len, obj_feature):
        q_output, s_hidden = self.qns_encoder(qas, qas_len, obj=obj_feature)
        qns_last_hidden = torch.squeeze(s_hidden)

        q_output, v_output = self.co_attn(q_output, v_output)

        adj = self.adj_learner(q_output, v_output)
        q_v_inputs = torch.cat((q_output, v_output), dim=1)
        q_v_output = self.gcn(q_v_inputs, adj)

        local_attn = self.gcn_atten_pool(q_v_output)
        local_out = torch.sum(q_v_output * local_attn, dim=1)

        global_out = self.global_fusion((qns_last_hidden, v_last_hidden))


        out = self.fusion((global_out, local_out)).squeeze()

        return out
