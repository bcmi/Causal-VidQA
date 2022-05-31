import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import init
import numpy as np
import os

def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)

class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0.1, n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.rnn_dropout = nn.Dropout(rnn_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            self.embedding = nn.Linear(input_dim, dim_embed, bias=False)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)

        self.obj_embedding = nn.Linear(2048, dim_embed, bias=False)

        self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional)

        # init_modules(self.modules(), w_init="xavier_uniform")
        # nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        if not self.use_bert and os.path.exists(self.glove_embed):
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

    def forward(self, qns, qns_lengths, hidden=None, obj=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        qns_embed = self.embedding(qns)
        assert obj is not None
        obj_embed = self.obj_embedding(obj)
        qns_embed = qns_embed + obj_embed
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.rnn_dropout(output)
        hidden = self.rnn_dropout(hidden).squeeze()
        return output, hidden


class EncoderVid(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVid, self).__init__()
        self.dim_vid = dim_vid
        self.dim_app = 2048
        self.dim_motion = 4096
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_vid, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, vid_feats):

        self.rnn.flatten_parameters()
        foutput, fhidden = self.rnn(vid_feats)

        return foutput, fhidden


class EncoderVidSTVQA(nn.Module):
    def __init__(self, input_dim, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidSTVQA, self).__init__()
        self.input_dim = input_dim
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell


        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn1 = self.rnn_cell(input_dim, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.rnn2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, vid_feats):
        """
        Dual-layer LSTM
        """

        self.rnn1.flatten_parameters()

        foutput_1, fhidden_1 = self.rnn1(vid_feats)
        self.rnn2.flatten_parameters()
        foutput_2, fhidden_2 = self.rnn2(foutput_1)

        foutput = torch.cat((foutput_1, foutput_2), dim=2)
        fhidden = (torch.cat((fhidden_1[0], fhidden_2[0]), dim=0),
                   torch.cat((fhidden_1[1], fhidden_2[1]), dim=0))

        return foutput, fhidden


class EncoderVidCoMem(nn.Module):
    def __init__(self, dim_app, dim_motion, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidCoMem, self).__init__()
        self.dim_app = dim_app
        self.dim_motion = dim_motion
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn_app_l1 = self.rnn_cell(self.dim_app, dim_hidden, n_layers, batch_first=True,
                                        bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self.rnn_app_l2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                        bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.rnn_motion_l1 = self.rnn_cell(self.dim_motion, dim_hidden, n_layers, batch_first=True,
                                            bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self.rnn_motion_l2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                           bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, vid_feats):
        """
        two separate LSTM to encode app and motion feature
        :param vid_feats:
        :return:
        """
        vid_app = vid_feats[:, :, 0:self.dim_app]
        vid_motion = vid_feats[:, :, self.dim_app:]

        app_output_l1, app_hidden_l1 = self.rnn_app_l1(vid_app)
        app_output_l2, app_hidden_l2 = self.rnn_app_l2(app_output_l1)


        motion_output_l1, motion_hidden_l1 = self.rnn_motion_l1(vid_motion)
        motion_output_l2, motion_hidden_l2 = self.rnn_motion_l2(motion_output_l1)

        return app_output_l1, app_output_l2, motion_output_l1, motion_output_l2


class EncoderVidHGA(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidHGA, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell


        self.vid2hid = nn.Sequential(nn.Linear(self.dim_vid, dim_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))


        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight)


    def forward(self, vid_feats):
        """
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats_trans = self.vid2hid(vid_feats.view(-1, self.dim_vid))
        vid_feats = vid_feats_trans.view(batch_size, seq_len, -1)

        self.rnn.flatten_parameters()
        foutput, fhidden = self.rnn(vid_feats)

        return foutput, fhidden

class EncoderVidB2A(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidB2A, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell


        self.vid2hid = nn.Sequential(nn.Linear(self.dim_vid, dim_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))


        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight)


    def forward(self, app_feat, mot_feat):
        """
        """
        batch_size, seq_len, seq_len2, dim_vid = app_feat.size()

        app_feat_trans = self.vid2hid(app_feat.view(-1, self.dim_vid))
        app_feat = app_feat_trans.view(batch_size, seq_len*seq_len2, -1)

        mot_feat_trans = self.vid2hid(mot_feat.view(-1, self.dim_vid))
        mot_feat = mot_feat_trans.view(batch_size, seq_len, -1)

        self.rnn.flatten_parameters()
        app_output, _ = self.rnn(app_feat)
        mot_output, _ = self.rnn(mot_feat)

        return app_output, mot_output