import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from networks.Transformer import SingleSimpleAttention
from networks.GCN import AdjGenerator, GCN
import EncoderRNN
from block import fusions #pytorch >= 1.1.0


class B2A(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device, gcn_layer=1):
        """
        Bridge to Answer: Structure-aware Graph Interaction Network for Video Question Answering (CVPR 2021)
        """
        super(B2A, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        hidden_size = qns_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p

        self.q_input_ln = nn.LayerNorm(hidden_size*2, elementwise_affine=False)
        self.v_input_ln = nn.LayerNorm(hidden_size*2, elementwise_affine=False)

        self.co_attn_t2m_qv = SingleSimpleAttention(
            hidden_size*2, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)
        self.co_attn_t2a_qv = SingleSimpleAttention(
            hidden_size*2, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)
        self.co_attn_a2t_vv = SingleSimpleAttention(
            hidden_size*2, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)
        self.co_attn_t2m_vv = SingleSimpleAttention(
            hidden_size*2, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)
        self.co_attn_m2t_vv = SingleSimpleAttention(
            hidden_size*2, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)
        self.co_attn_t2a_vv = SingleSimpleAttention(
            hidden_size*2, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)

        self.adj_generator = AdjGenerator(hidden_size*2, hidden_size*2)

        self.gcn_v = GCN(
            hidden_size*2,
            hidden_size*2,
            hidden_size*2,
            num_layers=gcn_layer,
            dropout=input_dropout_p)
        
        self.gcn_t = GCN(
            hidden_size*2,
            hidden_size*2,
            hidden_size*2,
            num_layers=gcn_layer,
            dropout=input_dropout_p)

        self.output_layer = OutputUnitMultiChoices(hidden_size*2)

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
        if self.qns_encoder.use_bert:
            candidates = candidates.permute(1, 0, 2, 3)
        else:
            candidates = candidates.permute(1, 0, 2)

        obj_feature = obj_feature.permute(1, 0, 2, 3)
        cand_len = candidates_len.permute(1, 0)

        app_output, mot_output = self.vid_encoder(video_appearance_feat, video_motion_feat)
        app_output = self.v_input_ln(app_output)
        mot_output = self.v_input_ln(mot_output)

        ques_output, ques_hidden = self.qns_encoder(question, question_len, obj=obj_fea_q)
        ques_output = ques_output.reshape(ques_output.shape[0], ques_output.shape[1], -1)
        ques_output = self.q_input_ln(ques_output)
        ques_hidden = ques_hidden.permute(1, 0, 2).reshape(ques_output.shape[0], -1)

        q_v_emb = self.q2v_v2v(app_output, mot_output, ques_output, dep_adj_q)


        out = []
        for idx, qas in enumerate(candidates):
            qas_output, qas_hidden = self.qns_encoder(qas, cand_len[idx], obj=obj_feature[idx])
            qas_output = qas_output.reshape(qas_output.shape[0], qas_output.shape[1], -1)
            qas_output = self.q_input_ln(qas_output)
            qas_hidden = qas_hidden.permute(1, 0, 2).reshape(qas_output.shape[0], -1)
            qa_v_emb = self.q2v_v2v(app_output, mot_output, qas_output, dep_adj[:, idx])

            final_output = self.output_layer(q_v_emb, qa_v_emb, ques_hidden, qas_hidden)
            out.append(final_output)

        out = torch.stack(out, 0).transpose(1, 0).squeeze()
        _, predict_idx = torch.max(out, 1)

        return out, predict_idx

    def q2v_v2v(self, app_feat, mot_feat, txt_feat, txt_cont=None):
        app_adj = self.adj_generator(app_feat)
        mot_adj = self.adj_generator(mot_feat)
        txt_adj = self.adj_generator(txt_feat, adjacency=txt_cont)

        # question-to-visual
        app_hat = self.gcn_v(self.co_attn_t2a_qv(app_feat, txt_feat), app_adj)
        mot_hat = self.gcn_v(self.co_attn_t2m_qv(mot_feat, txt_feat), mot_adj)

        # visual-to-visual
        txt_a2t = self.gcn_t(self.co_attn_a2t_vv(txt_feat, app_hat), txt_adj) + txt_feat
        txt_m2t = self.gcn_t(self.co_attn_m2t_vv(txt_feat, mot_hat), txt_adj) + txt_feat
        app_v2v = self.co_attn_t2a_vv(app_hat, txt_m2t)
        mot_v2v = self.co_attn_t2a_vv(mot_hat, txt_a2t)

        return torch.cat([app_v2v.mean(dim=1), mot_v2v.mean(dim=1)], dim=-1)

class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.v_question_proj = nn.Linear(module_dim*2, module_dim)

        self.v_ans_candidates_proj = nn.Linear(module_dim*2, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, q_visual_embedding, a_visual_embedding, question_embedding, ans_candidates_embedding):
        q_visual_embedding = self.v_question_proj(q_visual_embedding)
        a_visual_embedding = self.v_ans_candidates_proj(a_visual_embedding)
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding, ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out