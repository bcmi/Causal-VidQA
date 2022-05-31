from networks import Embed_loss, EncoderRNN, CRN
from networks.VQAModel import EVQA, HCRN, CoMem, HME, HGA, B2A
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import time


class VideoQA():
    def __init__(self, vocab, train_loader, val_loader, test_loader, glove_embed, use_bert, checkpoint_path, model_type,
                 model_prefix, vis_step, lr_rate, batch_size, epoch_num, logger, args):
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.glove_embed = glove_embed
        self.use_bert = use_bert
        self.model_dir = checkpoint_path
        self.model_type = model_type
        self.model_prefix = model_prefix
        self.vis_step = vis_step
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.logger = logger
        self.args = args

    def build_model(self):

        vid_dim = self.args.vid_dim
        hidden_dim = self.args.hidden_dim
        word_dim = self.args.word_dim
        vocab_size = len(self.vocab)
        max_vid_len = self.args.max_vid_len
        max_vid_frame_len = self.args.max_vid_frame_len
        max_qa_len = self.args.max_qa_len
        spl_resolution = self.args.spl_resolution
        
        if self.model_type == 'EVQA' or self.model_type == 'BlindQA':
            #ICCV15, AAAI17
            vid_encoder = EncoderRNN.EncoderVid(vid_dim, hidden_dim, input_dropout_p=0.2, n_layers=1, rnn_dropout_p=0, bidirectional=False, rnn_cell='gru')
            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=1, input_dropout_p=0.2, rnn_dropout_p=0, bidirectional=False, rnn_cell='gru')

            self.model = EVQA.EVQA(vid_encoder, qns_encoder, self.device, self.model_type == 'BlindQA')
        
        elif self.model_type == 'CoMem':
            #CVPR18
            app_dim = 2048
            motion_dim = 2048
            vid_encoder = EncoderRNN.EncoderVidCoMem(app_dim, motion_dim, hidden_dim, input_dropout_p=0.2, bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=2, rnn_dropout_p=0.5, input_dropout_p=0.2, bidirectional=False, rnn_cell='gru')

            self.model = CoMem.CoMem(vid_encoder, qns_encoder, max_vid_len, max_qa_len, self.device)

        elif self.model_type == 'HME':
            #CVPR19
            app_dim = 2048
            motion_dim = 2048
            vid_encoder = EncoderRNN.EncoderVidCoMem(app_dim, motion_dim, hidden_dim, input_dropout_p=0.2, bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=2, rnn_dropout_p=0.5, input_dropout_p=0.2, bidirectional=False, rnn_cell='gru')

            self.model = HME.HME(vid_encoder, qns_encoder, max_vid_len, max_qa_len*2, self.device)

        elif self.model_type == 'HGA':
            #AAAI20
            vid_encoder = EncoderRNN.EncoderVidHGA(vid_dim, hidden_dim, input_dropout_p=0.3, bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=1, rnn_dropout_p=0, input_dropout_p=0.3, bidirectional=False, rnn_cell='gru')

            self.model = HGA.HGA(vid_encoder, qns_encoder, self.device)
        
        elif self.model_type == 'HCRN':
            #CVPR20
            vid_dim = vid_dim//2
            vid_encoder = CRN.EncoderVidCRN(max_vid_frame_len, max_vid_len, spl_resolution, vid_dim, hidden_dim)

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=1, rnn_dropout_p=0.2, input_dropout_p=0.3, bidirectional=False, rnn_cell='gru')

            self.model = HCRN.HCRN(vid_encoder, qns_encoder, self.device)

        elif self.model_type == 'B2A' or self.model_type == 'B2A2':
            #CVPR21
            vid_dim = vid_dim // 2
            vid_encoder = EncoderRNN.EncoderVidB2A(vid_dim, hidden_dim*2, input_dropout_p=0.3, bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=1,  rnn_dropout_p=0, input_dropout_p=0.3, bidirectional=True, rnn_cell='gru')

            self.model = B2A.B2A(vid_encoder, qns_encoder, self.device)
        


        params = [{'params':self.model.parameters()}]

        self.optimizer = torch.optim.Adam(params = params, lr=self.lr_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=5, verbose=True)

        self.model.to(self.device)
        self.criterion = Embed_loss.MultipleChoiceLoss().to(self.device)


    def save_model(self, epoch, acc, is_best=False):
        if not is_best:
            torch.save(self.model.state_dict(), osp.join(self.model_dir, self.model_type, self.model_prefix, 'model', '{}-{:.2f}.ckpt'
                                                     .format(epoch, acc)))
        else:
            torch.save(self.model.state_dict(), osp.join(self.model_dir, self.model_type, self.model_prefix, 'model', 'best.ckpt'))

    def resume(self, model_file):
        """
        initialize model with pretrained weights
        :return:
        """
        self.logger.info('Warm-start (or test) with model: {}'.format(model_file))
        model_dict = torch.load(model_file)
        new_model_dict = {}
        for k, v in self.model.state_dict().items():
            if k in model_dict:
                v = model_dict[k]
            else:
                pass
                # print(k)
            new_model_dict[k] = v
        self.model.load_state_dict(new_model_dict)


    def run(self, model_file, pre_trained=False):
        self.build_model()
        self.logger.info(self.model)
        best_eval_score = 0.0
        if pre_trained:
            self.resume(model_file)
            best_eval_score = self.eval(0)
            self.logger.info('Initial Acc {:.2f}'.format(best_eval_score))

        for epoch in range(0, self.epoch_num):
            train_loss, train_acc = self.train(epoch)
            eval_score = self.eval(epoch)
            eval_score_test = self.eval_t(epoch)
            self.logger.info("==>Epoch:[{}/{}][Train Loss: {:.4f}; Train acc: {:.2f}; Val acc: {:.2f}; Test acc: {:.2f}]".
                  format(epoch, self.epoch_num, train_loss, train_acc, eval_score, eval_score_test))
            self.scheduler.step(eval_score)
            self.save_model(epoch, eval_score)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                self.save_model(epoch, best_eval_score, True)

    def train(self, epoch):
        self.logger.info('==>Epoch:[{}/{}][lr_rate: {}]'.format(epoch, self.epoch_num, self.optimizer.param_groups[0]['lr']))
        self.model.train()
        total_step = len(self.train_loader)
        epoch_loss = 0.0
        prediction_list = []
        answer_list = []
        for iter, inputs in enumerate(self.train_loader):
            visual, can, ques, ans_id, qns_key = inputs
            app_inputs = visual[0].to(self.device)
            mot_inputs = visual[1].to(self.device)
            candidate = can[0].to(self.device)
            candidate_lengths = can[1]
            obj_fea_can = can[2].to(self.device)
            dep_adj_can = can[3].to(self.device)
            question = ques[0].to(self.device)
            ques_lengths = ques[1]
            obj_fea_q = ques[2].to(self.device)
            dep_adj_q = ques[3].to(self.device)
            ans_targets = ans_id.to(self.device)
            out, prediction = self.model(app_inputs, mot_inputs, candidate, candidate_lengths, obj_fea_can, dep_adj_can, question, ques_lengths, obj_fea_q, dep_adj_q)

            self.model.zero_grad()
            loss = self.criterion(out, ans_targets)
            if not torch.isnan(loss):
                loss.backward()
            else:
                print(out)
                print(ans_targets)
            self.optimizer.step()
            epoch_loss += loss.item()
            if iter % self.vis_step == 0:
                self.logger.info('\t[{}/{}] Training loss: {:.4f}'.format(iter, total_step, epoch_loss/(iter+1)))

            prediction_list.append(prediction)
            answer_list.append(ans_id)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers==ref_answers).numpy()
        print(len(ref_answers))

        return epoch_loss / total_step, acc_num*100.0 / len(ref_answers)


    def eval(self, epoch):
        self.logger.info('==>Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))
        self.model.eval()
        total_step = len(self.val_loader)
        acc_count = 0
        prediction_list = []
        answer_list = []
        with torch.no_grad():
            for iter, inputs in enumerate(self.val_loader):
                visual, can, ques, ans_id, qns_key = inputs
                app_inputs = visual[0].to(self.device)
                mot_inputs = visual[1].to(self.device)
                candidate = can[0].to(self.device)
                candidate_lengths = can[1]
                obj_fea_can = can[2].to(self.device)
                dep_adj_can = can[3].to(self.device)
                question = ques[0].to(self.device)
                ques_lengths = ques[1]
                obj_fea_q = ques[2].to(self.device)
                dep_adj_q = ques[3].to(self.device)
                out, prediction = self.model(app_inputs, mot_inputs, candidate, candidate_lengths, obj_fea_can, dep_adj_can, question, ques_lengths, obj_fea_q, dep_adj_q)

                prediction_list.append(prediction)
                answer_list.append(ans_id)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers == ref_answers).numpy()
        print(len(ref_answers))

        return acc_num*100.0 / len(ref_answers)

    def eval_t(self, epoch):
        self.logger.info('==>Epoch:[{}/{}][test stage]'.format(epoch, self.epoch_num))
        self.model.eval()
        total_step = len(self.test_loader)
        acc_count = 0
        prediction_list = []
        answer_list = []
        with torch.no_grad():
            for iter, inputs in enumerate(self.test_loader):
                visual, can, ques, ans_id, qns_key = inputs
                app_inputs = visual[0].to(self.device)
                mot_inputs = visual[1].to(self.device)
                candidate = can[0].to(self.device)
                candidate_lengths = can[1]
                obj_fea_can = can[2].to(self.device)
                dep_adj_can = can[3].to(self.device)
                question = ques[0].to(self.device)
                ques_lengths = ques[1]
                obj_fea_q = ques[2].to(self.device)
                dep_adj_q = ques[3].to(self.device)
                out, prediction = self.model(app_inputs, mot_inputs, candidate, candidate_lengths, obj_fea_can, dep_adj_can, question, ques_lengths, obj_fea_q, dep_adj_q)

                prediction_list.append(prediction)
                answer_list.append(ans_id)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers == ref_answers).numpy()
        print(len(ref_answers))

        return acc_num*100.0 / len(ref_answers)


    def predict(self, model_file, result_file, loader):
        """
        predict the answer with the trained model
        :param model_file:
        :return:
        """
        self.build_model()
        self.resume(model_file)

        self.model.eval()
        results = {}
        with torch.no_grad():
            for iter, inputs in enumerate(loader):
                visual, can, ques, ans_id, qns_key = inputs
                app_inputs = visual[0].to(self.device)
                mot_inputs = visual[1].to(self.device)
                candidate = can[0].to(self.device)
                candidate_lengths = can[1]
                obj_fea_can = can[2].to(self.device)
                dep_adj_can = can[3].to(self.device)
                question = ques[0].to(self.device)
                ques_lengths = ques[1]
                obj_fea_q = ques[2].to(self.device)
                dep_adj_q = ques[3].to(self.device)
                out, prediction = self.model(app_inputs, mot_inputs, candidate, candidate_lengths, obj_fea_can, dep_adj_can, question, ques_lengths, obj_fea_q, dep_adj_q)
                
                prediction = prediction.data.cpu().numpy()
                ans_id = ans_id.numpy()
                for qid, pred, ans in zip(qns_key, prediction, ans_id):
                    results[qid] = {'prediction': int(pred), 'answer': int(ans)}

        print(len(results))
        print(result_file)
        save_file(results, result_file)
