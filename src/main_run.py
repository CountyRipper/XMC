from argparse import ArgumentParser
import os
import re
from combine import get_combine_bi_list, get_combine_list
from rank import rank_bi,rank
from rank_training import rank_train

from trainer_kp import modeltrainer
from utils.p_at_1 import p_at_k


def run(args:ArgumentParser):
    print(args)
    args.modelname = 'bart-large'
    args.outputmodel = 'bart_save'
    args.checkdir = 'bart_check'
    trainer = modeltrainer(args)
    #define affix
    if re.match("\w*bart\w*",args.modelname,re.I):
        affix = 'ba'
    elif re.match("\w*pegasus\w*",args.modelname,re.I):
        affix = 'pega'
    elif re.match("\w*T5\w*",args.modelname,re.I):
        affix = 't5'
    elif re.match("\w*kaybart\w*",args.modelname,re.I):
        affix = 'kb'
    else: affix = ''
    args.add_argument('--affix',type=str,default=affix)
    
    if args.istrain:
        #run finetune
        trainer.train()
    if args.is_pred_trn:
        trainer.predicting(args.outputmodel,args.train_texts,"train_pred"+"_"+affix+".txt")
    if args.is_pred_tst:
        trainer.predicting(args.outputmodel,args.test_texts,"test_pred"+"_"+affix+".txt")
    if args.iscombine:
        if args.combine_model=='cross-encoder':
            if os.path.exists(os.path.join(args.datadir,"train_pred"+"_"+affix+".txt")):
                get_combine_list(args.datadir,"train_pred"+"_"+affix+".txt",
                                 args.all_labels,"train_combine_"+affix+".txt") #train_pred_fix.txt
            if os.path.exists(os.path.join(args.datadir,"test_pred"+"_"+affix+".txt")):
                get_combine_list(args.datadir,"test_pred"+"_"+affix+".txt",
                                 args.all_labels,"test_combine_"+affix+".txt") #test_pred_fix.txt
        else:
            if os.path.exists(os.path.join(args.datadir,"train_pred"+"_"+affix+".txt")):
                get_combine_bi_list(args.datadir,"train_pred"+"_"+affix+".txt",
                                    args.all_labels,"train_combine_"+affix+".txt") #train_pred_fix.txt
            if os.path.exists(os.path.join(args.datadir,"test_pred"+"_"+affix+".txt")):
                get_combine_bi_list(args.datadir,"test_pred"+"_"+affix+".txt",
                                    args.all_labels,"test_combine_"+affix+".txt") #test_pred_fix.txt    
    args.is_rank_train=True
    args.rank_model = "all-MiniLM-L6-v2"
    if re.match("\w*cross-encoder\w*",args.rank_model,re.I):
        affix1 = 'cr'
    else: affix1 = 'bi' 
        
    if args.is_rank_train:
        rank_train(args.datadir,args.rank_model,args.train_texts,"train_combine_"+affix+".txt",args.train_labels,args.rankmodel_save)
    if args.is_ranking :
        if affix1 =='cr':
            rank(args.datadir,args.test_texts,"test_combine_"+affix+".txt",args.model_save,"test_ranked_"+affix+affix1+".txt")
        else:
            rank_bi(args.datadir,args.test_texts,"test_combine_"+affix+".txt",args.model_save,"test_ranked_"+affix+affix1+".txt")
    p_at_k(args.datadir,args.test_labels,"test_ranked_"+affix+affix1+".txt")    
    
        


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all_labels',type=str,default='all_labels.txt')
    parser.add_argument('--test_labels',type=str,default='test_labels.txt')
    parser.add_argument('--train_labels',type=str,default="train_labels.txt")
    parser.add_argument('--test_texts',type=str,default="test_texts.txt")
    parser.add_argument('--train_texts',type=str,default="train_texts.txt")
    # finetune args
    parser.add_argument('--istrain',type=bool,default=True,
                        help="whether run finteune processing")
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='number of batch size for training')
    parser.add_argument('-e', '--epoch', type=int, default=5,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--modelname', type=str,default='pegasus',
                        help='modelname ')
    parser.add_argument('--datadir', type=str, default='./dataset/EUR-Lex/',
                        help='dataset_dir')
    parser.add_argument('--checkdir', type=str, default='pegasus_check',
                        help='path to trained model to save')
    parser.add_argument('--outputmodel',type=str,default='pegasus_save',
                        help="fine-tune model save dir")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=44,
                        help='random seed (default: 1)')
    #perdicting args
    parser.add_argument('--is_pred_trn',type=bool,default=True,
                        help="Whether run predicting training dataset")
    parser.add_argument('--is_pred_tst',type=bool,default=True,
                        help="Whether run predicting testing dataset")
    parser.add_argument('--top_k',type=int,default=10)
    parser.add_argument('--')

    #combine part
    parser.add_argument('--iscombine',type=bool,default=True,
                        help="Whether run combine")
    parser.add_argument('--combine_model',type=str,default='cross-encoder')

    parser.add_argument('--combine_testdir',type=str,default="test_pred.txt")
    parser.add_argument('--combine_traindir',type=str,default="train_pred.txt")
    parser.add_argument('--combine_testout',type=str,default="test_combine_labels.txt")
    parser.add_argument('--combine_trainout',type=str,default="train_combine_labels.txt")
    #rank part
    parser.add_argument('--is_rank_train',type=bool,default=True)
    parser.add_argument('--rank_model',type=str,default='cross-encoder/stsb-roberta-base')
    parser.add_argument('--rank_batch',type=int,default=128)
    parser.add_argument('--rankmodel_save',type=str,default='cr_en')
    parser.add_argument('--rank_textdir',type=str,default='train_texts.txt')
    parser.add_argument('--is_ranking',type=bool,default=True)
    args = parser.parse_args()
    run(args)
