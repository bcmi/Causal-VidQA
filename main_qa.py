from videoqa import *
from dataset import VidQADataset, Vocabulary
from torch.utils.data import Dataset, DataLoader
from utils import *
import argparse
import eval_mc
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):

    mode = args.mode
    if mode == 'train':
        batch_size = args.batch_size
        num_worker = 8
    else:
        batch_size = 32
        num_worker = 8

    feature_path = args.feature_path
    text_feature_path = args.text_feature_path
    data_path = args.data_path
    train_split_path = osp.join(args.split_path, 'train.pkl')
    valid_split_path = osp.join(args.split_path, 'valid.pkl')
    test_split_path = osp.join(args.split_path, 'test.pkl')
    qtype=args.qtype
    max_qa_len = args.max_qa_len

    vocab = pkload(osp.join(text_feature_path, 'qa_vocab.pkl'))

    glove_embed = osp.join(text_feature_path, 'glove.840B.300d.npy')
    use_bert = args.use_bert
    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    model_prefix= args.model_prefix

    vis_step = args.vis_step
    lr_rate = args.lr_rate
    epoch_num = args.epoch_num

    if not osp.exists(osp.join(checkpoint_path, model_type, model_prefix)):
        os.makedirs(osp.join(checkpoint_path, model_type, model_prefix))
    if not osp.exists(osp.join(checkpoint_path, model_type, model_prefix, 'model')):
        os.makedirs(osp.join(checkpoint_path, model_type, model_prefix, 'model'))
    logger = make_logger(osp.join(checkpoint_path, model_type, model_prefix, 'log'))

    train_set = VidQADataset(feature_path=feature_path, text_feature_path=text_feature_path, split_path=train_split_path, data_path=data_path, use_bert=use_bert, vocab=vocab, qtype=qtype, max_length=max_qa_len)
    valid_set = VidQADataset(feature_path=feature_path, text_feature_path=text_feature_path, split_path=valid_split_path, data_path=data_path, use_bert=use_bert, vocab=vocab, qtype=qtype, max_length=max_qa_len)
    test_set = VidQADataset(feature_path=feature_path, text_feature_path=text_feature_path, split_path=test_split_path, data_path=data_path, use_bert=use_bert, vocab=vocab, qtype=qtype, max_length=max_qa_len)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker)
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker)
    
    vqa = VideoQA(vocab, train_loader, valid_loader, test_loader, glove_embed, use_bert, checkpoint_path, model_type, model_prefix,
                  vis_step, lr_rate, batch_size, epoch_num, logger, args)

    if mode != 'train':
        model_file = osp.join(args.checkpoint_path, model_type, model_prefix, 'model', 'best.ckpt')
        result_file1 = args.result_file.format(model_type, model_prefix, 'valid')
        result_file2 = args.result_file.format(model_type, model_prefix, 'test')
        vqa.predict(model_file, result_file1, vqa.val_loader)
        vqa.predict(model_file, result_file2, vqa.test_loader)
        print('Validation set')
        eval_mc.main(result_file1, qtype=args.qtype)
        print('Test set')
        eval_mc.main(result_file2, qtype=args.qtype)
    else:
        model_file = osp.join(model_type, model_prefix, 'model', '0-00.00.ckpt')
        vqa.run(model_file, pre_trained=False)
        model_file = osp.join(args.checkpoint_path, model_type, model_prefix, 'model', 'best.ckpt')
        result_file1 = args.result_file.format(model_type, model_prefix, 'valid')
        result_file2 = args.result_file.format(model_type, model_prefix, 'test')
        vqa.predict(model_file, result_file1, vqa.val_loader)
        vqa.predict(model_file, result_file2, vqa.test_loader)
        print('Validation set')
        eval_mc.main(result_file1, qtype=args.qtype)
        print('Test set')
        eval_mc.main(result_file2, qtype=args.qtype)

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, 
                        help='gpu device id')
    parser.add_argument('--mode', type=str, default='train', 
                        help='train or val')
    parser.add_argument('--feature_path', type=str, default='', 
                        help='path to load visual feature')
    parser.add_argument('--text_feature_path', type=str, default='', 
                        help='path to load text feature')
    parser.add_argument('--data_path', type=str, default='', 
                        help='path to load original data')
    parser.add_argument('--split_path', type=str, default='', 
                        help='path for train/valid/test split')
    parser.add_argument('--use_bert', action='store_true', 
                        help='whether use bert embedding')
    parser.add_argument('--checkpoint_path', type=str, default='', 
                        help='path to save training model and log')
    parser.add_argument('--model_type', type=str, default='HGA', 
                        help='(B2A, EVQA, CoMem, HME, HGA, HCRN)')
    parser.add_argument('--model_prefix', type=str, default='debug', 
                        help='detail model info')
    parser.add_argument('--result_file', type=str, default='', 
                        help='where to save processed results')

    parser.add_argument('--vid_dim', type=int, default=4096, 
                        help='number of dim for video features')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                        help='number of dim for hidden feature')
    parser.add_argument('--word_dim', type=int, default=300, 
                        help='number of dim for word feature')
    parser.add_argument('--max_vid_len', type=int, default=8, 
                        help='number of max length for video clips')
    parser.add_argument('--max_vid_frame_len', type=int, default=16, 
                        help='number of max length for frames in each video clip')
    parser.add_argument('--max_qa_len', type=int, default=40, 
                        help='number of max length for question and answer')
    parser.add_argument('--vis_step', type=int, default=100, 
                        help='number of step to print the training info')
    parser.add_argument('--epoch_num', type=int, default=30, 
                        help='number of epoch to train model')
    parser.add_argument('--lr_rate', type=float, default=1e-4, 
                        help='learning rate')
    parser.add_argument('--qtype', type=int, default=-1, 
                        help='question type in VVCR dataset')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size')
    parser.add_argument('--gcn_layer', type=int, default=1, 
                        help='gcn layer')
    parser.add_argument('--spl_resolution', type=int, default=16, 
                        help='spl_resolution')
    args = parser.parse_args()

    main(args)