from locale import AM_STR
import torch
import torch.nn as nn


class EVQA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device, blind=False):
        super(EVQA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        self.blind = blind
        self.FC = nn.Linear(qns_encoder.dim_hidden, 1)

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
        if self.blind:
            obj_feature[:] = 0
        cand_len = candidates_len.permute(1, 0)
        out = []
        for idx, qnsans in enumerate(candidates):
            encoder_out = self.vq_encoder(vid_feats, qnsans, cand_len[idx], question_len, obj_feature[idx])
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)

        _, predict_idx = torch.max(out, 1)

        return out, predict_idx

    def vq_encoder(self, vid_feats, qnsans, qnsans_len, qns_len, obj_feature):

        qmask = torch.zeros(qnsans.shape[0], qnsans.shape[1], dtype=qnsans.dtype, device=qnsans.device) # bs, maxlen
        amask = torch.zeros(qnsans.shape[0], qnsans.shape[1], dtype=qnsans.dtype, device=qnsans.device) # bs, maxlen

        for idx in range(qmask.shape[0]):
            qmask[idx, :qns_len[idx]] = 1
            amask[idx, qns_len[idx]:qnsans_len[idx]] = 1

        if len(qnsans.shape) == 2:
            qns = qnsans*qmask
            ans = qnsans*amask
        elif len(qnsans.shape) == 3:
            qns = qnsans*qmask.unsqueeze(-1)
            ans = qnsans*amask.unsqueeze(-1)
        
        obj_feature_q = obj_feature*qmask.unsqueeze(-1)
        obj_feature_a = obj_feature*amask.unsqueeze(-1)

        _, vid_hidden = self.vid_encoder(vid_feats)
        _, qs_hidden = self.qns_encoder(qns, qns_len, obj=obj_feature_q)
        _, as_hidden = self.qns_encoder(ans, qnsans_len, obj=obj_feature_a)

        vid_embed = vid_hidden.squeeze()
        qs_embed = qs_hidden.squeeze()
        as_embed = as_hidden.squeeze()

        if self.blind:
            fuse = qs_embed + as_embed
        else:
            fuse = qs_embed + as_embed + vid_embed

        outputs = self.FC(fuse).squeeze()

        return outputs