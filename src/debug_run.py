from argparse import ArgumentParser
from main_run import run
#from utils.p_at_1 import p_at_k

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./dataset/EUR-Lex/',
                        help='dataset_dir')
    parser.add_argument('--test_json',type=str,default='test_finetune.json')
    parser.add_argument('--train_json',type=str,default='train_finetune.json')
    parser.add_argument('--all_labels',type=str,default='all_labels.txt')
    parser.add_argument('--test_labels',type=str,default='test_labels.txt')
    parser.add_argument('--train_labels',type=str,default="train_labels.txt")
    parser.add_argument('--test_texts',type=str,default="test_texts.txt")
    parser.add_argument('--train_texts',type=str,default="train_texts.txt")
    # finetune args
    parser.add_argument('--istrain',type=int,default=1,
                        help="whether run finteune processing")
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='number of batch size for training')
    parser.add_argument('-e', '--t2t_epoch', type=int, default=5,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--modelname', type=str,default='bart',
                        help='modelname ')
    parser.add_argument('--affix1',type=str,default="")
    parser.add_argument('--affix2',type=str,default="")
    parser.add_argument('--checkdir', type=str, default='bart_check',
                        help='path to trained model to save')
    parser.add_argument('--outputmodel',type=str,default='bart_save',
                        help="fine-tune model save dir")
    parser.add_argument('--t2t_lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=44,
                        help='random seed (default: 1)')
    #perdicting args
    parser.add_argument('--is_pred_trn',type=int,default=1,
                        help="Whether run predicting training dataset")
    parser.add_argument('--is_pred_tst',type=int,default=1,
                        help="Whether run predicting testing dataset")
    parser.add_argument('--top_k',type=int,default=10)
    parser.add_argument('--data_size',type=int,default=12)
    #combine part
    parser.add_argument('--iscombine',type=int,default=1,
                        help="Whether run combine")
    parser.add_argument('--combine_model',type=str,default='bi-encoder')
    parser.add_argument('--combine_model_name',type=str,default='all-MiniLM-L6-v2')
    parser.add_argument('--combine_testdir',type=str,default="test_pred.txt")
    parser.add_argument('--combine_traindir',type=str,default="train_pred.txt")
    parser.add_argument('--combine_testout',type=str,default="test_combine_labels.txt")
    parser.add_argument('--combine_trainout',type=str,default="train_combine_labels.txt")
    #rank part
    parser.add_argument('--is_rank_train',type=int,default=1)
    parser.add_argument('--rank_model',type=str,default='all-MiniLM-L6-v2')
    parser.add_argument('--rank_batch',type=int,default=64)
    parser.add_argument('--rank_epoch',type=int,default=4)
    parser.add_argument('--rankmodel_save',type=str,default='ba_bi_bi64')
    parser.add_argument('--rank_textdir',type=str,default='train_texts.txt')
    parser.add_argument('--is_ranking',type=int,default=1)
    parser.add_argument('--rank_is_trained',type=int,default=1)
    args = parser.parse_args()
    args.datadir = './dataset/Wiki10-31K/'
    args.istrain=0
    args.is_pred_trn=0
    args.is_pred_tst=0
    args.iscombine=0
    args.is_rank_train=0
    args.is_ranking=1
    args.combine_model='bi-encoder'
    args.combine_model_name='all-MiniLM-L6-v2'
    args.modelname='BART'
    args.outputmodel='bart_save'
    args.batch_size=2
    args.t2t_epoch=3
    args.t2t_lr=5e-5
    args.checkdir='bart_check'
    args.data_size=4
    args.rank_model='princeton-nlp/unsup-simcse-roberta-base'
    args.rank_batch=16
    args.rank_epoch=3
    args.rankmodel_save='princeton-nlp/unsup-simcse-roberta-base'
    args.rank_is_trained = 0
    run(args)
