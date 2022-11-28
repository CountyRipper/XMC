from argparse import ArgumentParser
import datetime,time
import os
import re

import torch
from combine import get_combine_bi_list, get_combine_list
from utils.premethod import clean_set
from rank import rank_bi,rank
from rank_training import rank_train

from trainer_kp import modeltrainer
from utils.p_at_1 import p_at_k

def run(args:ArgumentParser):
    
    datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/','./dataset/AmazonCat-13K/',
               './dataset/AmazonCat-13K-10/','./dataset/Wiki500K-20/']

    models = {'pega':1,'bart':0,'kb':0}
    # args.datadir = datadir[1]
    # args.modelname = 'pegausu-large'
    # args.outputmodel = 'pegasus_save'
    # args.checkdir = 'pegasus_check'
    # args.batch_size = 2
    # args.epoch = 5
    # args.istrain = False
    # args.is_pred_trn = False
    # args.is_pred_tst = False
    # args.iscombine = False
    # args.combine_model = 'bi-encoder'
    # args.is_rank_train  =False
    # args.is_ranking = True
    # args.rank_model = 'cross-encoder/stsb-roberta-base'
    # args.rankmodel_save = 'cr_en'
    affix1 = 'pega'
    if re.match("\w*bart\w*",args.modelname,re.I):
        affix1 = 'ba'
        if re.match("\w*bart-large\w*",args.modelname,re.I):
            affix1 = 'bal'
    elif re.match("\w*pegasus\w*",args.modelname,re.I):
        affix1 = 'pega'
    elif re.match("\w*T5\w*",args.modelname,re.I):
        affix1 = 't5'
        if re.match("\w*T5-large\w*",args.modelname,re.I):
            affix1='t5l'
    elif re.match("\w*kaybart\w*",args.modelname,re.I):
        affix1 = 'kb'
    args.affix1=affix1
    print(args)
    start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time_stap0 = time.process_time()
    trainer = modeltrainer(args)
    model_time1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time_stap1 = time.process_time()
    
    #define affix
    if args.istrain:
        #run finetune
        torch.cuda.empty_cache()
        trainer.train()   
        model_time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap2 = time.process_time()
    
    #args.top_k = 10
    #args.data_size = 14
    if args.is_pred_trn:
        torch.cuda.empty_cache()
        trainer.predicting(args.outputmodel,args.train_json,"train_pred"+"_"+affix1+".txt")    
        clean_set(args.datadir+'res',"train_pred"+"_"+affix1+".txt")
        model_time3 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap3 = time.process_time()
    if args.is_pred_tst:
        torch.cuda.empty_cache()
        trainer.predicting(args.outputmodel,args.test_json,"test_pred"+"_"+affix1+".txt")    
        clean_set(args.datadir+'res',"test_pred"+"_"+affix1+".txt")
        model_time4 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap4 = time.process_time()
    affix2='bi' #默认c r
    if args.iscombine:
        if args.combine_model=='cross-encoder':
            affix2='cr'
            if os.path.exists(os.path.join(args.datadir,'res',"train_pred"+"_"+affix1+".txt")):
                get_combine_list(args.datadir,"train_pred"+"_"+affix1+affix2+".txt",
                                 args.all_labels,"train_combine_labels_"+affix1+affix2+".txt") #train_pred_fix.txt
            if os.path.exists(os.path.join(args.datadir,'res',"test_pred"+"_"+affix1+".txt")):
                get_combine_list(args.datadir,"test_pred"+"_"+affix1+affix2+".txt",
                                 args.all_labels,"test_combine_labels_"+affix1+affix2+".txt") #test_pred_fix.txt
        else: # using bi
            affix2='bi'
            if os.path.exists(os.path.join(args.datadir,'res',"train_pred"+"_"+affix1+".txt")):
                get_combine_bi_list(args.datadir,"train_pred"+"_"+affix1+".txt",
                                    args.all_labels,"train_combine_labels_"+affix1+affix2+".txt") #train_pred_fix.txt
            if os.path.exists(os.path.join(args.datadir,'res',"test_pred"+"_"+affix1+".txt")):
                get_combine_bi_list(args.datadir,"test_pred"+"_"+affix1+".txt",
                                    args.all_labels,"test_combine_labels_"+affix1+affix2+".txt") #test_pred_fix.txt    
        model_time5 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap5 = time.process_time()
    #args.rank_model = "all-MiniLM-L6-v2"
    #args.rank_model = ""
    affix3 = 'cr'
    if re.match("\w*cross-encoder\w*",args.rank_model,re.I):
        affix3 = 'cr'
    else: affix3 = 'bi' 
    
    if args.is_rank_train:
        rank_train(args.datadir,args.rank_model,args.train_texts,"train_combine_labels_"+affix1+affix2+".txt",args.train_labels,args.rankmodel_save)
        model_time6 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap6 = time.process_time()
    if args.is_ranking :
        if affix3 =='cr':
            rank(args.datadir,args.test_texts,"test_combine_labels_"+affix1+affix2+".txt",args.rankmodel_save,"test_ranked_labels_"+affix1+affix2+affix3+".txt")
        else:
            rank_bi(args.datadir,args.test_texts,"test_combine_labels_"+affix1+affix2+".txt",args.model_save,"test_ranked_labels_"+affix1+affix2+affix3+".txt")
        model_time7 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap7 = time.process_time()
    with open('./log/run_time.txt','a+') as w:
        if model_time1:
            w.write('start:'+model_time1+'\n')
        if model_time2:
            w.write('text2text model train endtime:'+model_time2+'\n')
        if model_time3:
            w.write('pre_trn endtime:'+model_time3+'\n')
        if model_time4:
            w.write('pre_tst endtime:'+model_time4+'\n')
        if model_time5:
            w.write('combine endtime:'+model_time5+'\n')
        if model_time6:
            w.write('rank model train endtime:'+model_time6+'\n')
        if model_time7:
            w.write('ranking endtime:'+model_time7+'\n')
    p_at_k(args.datadir,args.test_labels,"test_ranked_labels_"+affix1+affix2+affix3+".txt",args.datadir+"res_"+affix1+affix2+affix3+".txt")    

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
    parser.add_argument('-e', '--epoch', type=int, default=5,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--modelname', type=str,default='pegasus',
                        help='modelname ')
    parser.add_argument('--affix1',type=str,default="")
    parser.add_argument('--affix2',type=str,default="")
    parser.add_argument('--checkdir', type=str, default='pegasus_check',
                        help='path to trained model to save')
    parser.add_argument('--outputmodel',type=str,default='pegasus_save',
                        help="fine-tune model save dir")
    parser.add_argument('--lr', type=float, default=2e-5,
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
    parser.add_argument('--combine_model',type=str,default='cross-encoder')
    parser.add_argument('--combine_testdir',type=str,default="test_pred.txt")
    parser.add_argument('--combine_traindir',type=str,default="train_pred.txt")
    parser.add_argument('--combine_testout',type=str,default="test_combine_labels.txt")
    parser.add_argument('--combine_trainout',type=str,default="train_combine_labels.txt")
    #rank part
    parser.add_argument('--is_rank_train',type=int,default=1,)
    parser.add_argument('--rank_model',type=str,default='cross-encoder/stsb-roberta-base')
    parser.add_argument('--rank_batch',type=int,default=128)
    parser.add_argument('--rankmodel_save',type=str,default='cr_en')
    parser.add_argument('--rank_textdir',type=str,default='train_texts.txt')
    parser.add_argument('--is_ranking',type=int,default=1)
    args = parser.parse_args()
    run(args)
