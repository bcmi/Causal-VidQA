import numpy as np
from torch.nn import functional as F

import itertools

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.module import Module

class HCRN(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device):
        super(HCRN, self).__init__()
        """
        Hierarchical Conditional Relation Networks for Video Question Answering (CVPR2020)
        """
        self.qns_encoder = qns_encoder
        self.vid_encoder = vid_encoder
        hidden_size = vid_encoder.dim_hidden
        self.feature_aggregation = FeatureAggregation(hidden_size)

        self.output_unit = OutputUnitMultiChoices(module_dim=hidden_size)

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
        batch_size = candidates.size(0)
        if self.qns_encoder.use_bert:
            cand = candidates.permute(1, 0, 2, 3)  # for BERT
        else:
            cand = candidates.permute(1, 0, 2)
        cand_len = candidates_len.permute(1, 0)
        out = list()
        _, question_embedding = self.qns_encoder(question, question_len, obj=obj_fea_q)
        visual_embedding = self.vid_encoder(video_appearance_feat, video_motion_feat, question_embedding)
        q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)
        for idx, qas in enumerate(cand):
            _, qas_embedding = self.qns_encoder(qas, cand_len[idx], obj=obj_feature[:, idx])
            qa_visual_embedding = self.feature_aggregation(qas_embedding, visual_embedding)
            encoder_out = self.output_unit(q_visual_embedding, question_embedding, qa_visual_embedding, qas_embedding)
            out.append(encoder_out)
        out = torch.stack(out, 0).transpose(1, 0).squeeze()
        _, predict_idx = torch.max(out, 1)
        return out, predict_idx

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out