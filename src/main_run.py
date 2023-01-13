from argparse import ArgumentParser
import datetime,time
import os
import re

import torch
from combine import combine_clean, get_combine_bi_list, get_combine_list, get_combine_simcse
from utils.premethod import clean_set, save_time, p_at_k
from rank import rank_bi,rank,rank_simcse
from rank_training import rank_train
from model.rank_model import Rank_model
from trainer_kp import modeltrainer
#from utils.p_at_1 import p_at_k

def run(args:ArgumentParser):
    
    datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/','./dataset/AmazonCat-13K/',
               './dataset/AmazonCat-13K-10/','./dataset/Wiki500K-20/']

    models = {'pega':1,'bart':0,'kb':0}
    model_time1=None
    model_time2=None
    model_time3=None
    model_time4=None
    model_time5=None
    model_time6=None
    model_time7=None
    try:
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
        save_time(start,args.datadir+'timelog.txt','start')
        trainer = modeltrainer(args)
        model_time1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap1 = time.process_time()
        save_time(model_time1,args.datadir+'timelog.txt','text2text model train start')

        #define affix
        if args.istrain:
            #run finetune
            torch.cuda.empty_cache()
            trainer.train()   
            model_time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time_stap2 = time.process_time()
            save_time(model_time2,args.datadir+'timelog.txt','text2text model train end')
        #args.top_k = 10
        #args.data_size = 14
        if args.is_pred_trn:
            torch.cuda.empty_cache()
            trainer.predicting(args.outputmodel,args.train_json,"train_pred"+"_"+affix1+".txt")    
            clean_set(args.datadir+'res',"train_pred"+"_"+affix1+".txt")
            model_time3 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time_stap3 = time.process_time()
            save_time(model_time3,args.datadir+'timelog.txt','text2text model train pred end')
        if args.is_pred_tst:
            torch.cuda.empty_cache()
            trainer.predicting(args.outputmodel,args.test_json,"test_pred"+"_"+affix1+".txt")    
            clean_set(args.datadir+'res',"test_pred"+"_"+affix1+".txt")
            model_time4 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time_stap4 = time.process_time()

            save_time(model_time4,args.datadir+'timelog.txt','text2text model test pred end')
        affix2='bi' #默认cr
        
        if args.combine_model=='cross-encoder':
            affix2='cr'
            if args.iscombine:
                if os.path.exists(os.path.join(args.datadir,'res',"train_pred"+"_"+affix1+".txt")):
                    get_combine_list(args.datadir,"train_pred"+"_"+affix1+affix2+".txt",
                                 args.all_labels,args.combine_model_name,"train_combine_labels_"+affix1+affix2+".txt") #train_pred_fix.txt
                if os.path.exists(os.path.join(args.datadir,'res',"test_pred"+"_"+affix1+".txt")):
                    get_combine_list(args.datadir,"test_pred"+"_"+affix1+affix2+".txt",
                                 args.all_labels,args.combine_model_name,"test_combine_labels_"+affix1+affix2+".txt") #test_pred_fix.txt
        elif args.combine_model =='bi-encoder': # using bi
            affix2='bi'
            if args.iscombine:
                if os.path.exists(os.path.join(args.datadir,'res',"train_pred"+"_"+affix1+".txt")):
                    get_combine_bi_list(args.datadir,"train_pred"+"_"+affix1+".txt",
                                    args.all_labels,args.combine_model_name,"train_combine_labels_"+affix1+affix2+".txt") #train_pred_fix.txt
                if os.path.exists(os.path.join(args.datadir,'res',"test_pred"+"_"+affix1+".txt")):
                    get_combine_bi_list(args.datadir,"test_pred"+"_"+affix1+".txt",
                                    args.all_labels,args.combine_model_name,"test_combine_labels_"+affix1+affix2+".txt",) #test_pred_fix.txt    
        elif args.combine_model =='simcse':
            affix2 = 'sim'
            if args.iscombine:
                if os.path.exists(os.path.join(args.datadir,'res',"train_pred"+"_"+affix1+".txt")):
                    get_combine_simcse(args.datadir,"train_pred"+"_"+affix1+".txt",
                                    args.all_labels,args.combine_model_name,"train_combine_labels_"+affix1+affix2+".txt") #train_pred_fix.txt
                if os.path.exists(os.path.join(args.datadir,'res',"test_pred"+"_"+affix1+".txt")):
                    get_combine_simcse(args.datadir,"test_pred"+"_"+affix1+".txt",
                                    args.all_labels,args.combine_model_name,"test_combine_labels_"+affix1+affix2+".txt") #test_pred_fix.txt    
        else:
            print('cl'+'\n')
            affix2='cl'
            if os.path.exists(os.path.join(args.datadir,'res',"train_pred"+"_"+affix1+".txt")):
                combine_clean(args.datadir,"train_pred"+"_"+affix1+".txt",
                                    args.all_labels,"train_combine_labels_"+affix1+affix2+".txt") #train_pred_fix.txt
            if os.path.exists(os.path.join(args.datadir,'res',"test_pred"+"_"+affix1+".txt")):
                combine_clean(args.datadir,"test_pred"+"_"+affix1+".txt",
                                    args.all_labels,"test_combine_labels_"+affix1+affix2+".txt") #test_pred_fix.txt
        model_time5 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_stap5 = time.process_time()
        save_time(model_time5,args.datadir+'timelog.txt','combine end')

        affix3 = 'cr'
        if 'cross-encoder' in args.rank_model:
            affix3 = 'cr'
        elif 'simcse' in args.rank_model:
            affix3 = 'sim'  
        else: affix3 = 'bi'
        if args.is_rank_train:
            rank_train(args.datadir,args.rank_model,args.train_texts,"train_combine_labels_"+affix1+affix2+".txt",args.train_labels,args.rankmodel_save,args.rank_batch,args.rank_epoch)
            model_time6 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time_stap6 = time.process_time()
            save_time(model_time6,args.datadir+'timelog.txt','rank train end')
        if args.is_ranking :
            if affix3 =='cr':
                rank(args.datadir,args.test_texts,"test_combine_labels_"+affix1+affix2+".txt",args.rankmodel_save,"test_ranked_labels_"+affix1+affix2+affix3+".txt")
            elif affix3 =='sim':
                rank_simcse(args.datadir,args.test_texts,"test_combine_labels_"+affix1+affix2+".txt",args.rankmodel_save,args.rank_is_trained,"test_ranked_labels_"+affix1+affix2+affix3+".txt")
            else:
                rank_bi(args.datadir,args.test_texts,"test_combine_labels_"+affix1+affix2+".txt",args.rankmodel_save,"test_ranked_labels_"+affix1+affix2+affix3+".txt")
            model_time7 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time_stap7 = time.process_time()
            save_time(model_time7,args.datadir+'timelog.txt','rank  end')
        # res
        res = None
        res = p_at_k(args.datadir,args.test_labels,"test_ranked_labels_"+affix1+affix2+affix3+".txt",args.datadir+"res_"+affix1+affix2+affix3+".txt")
    finally:
        with open('./log/run_time.txt','a+') as w:
            if model_time1:
                w.write('start: '+model_time1+'\n')
                w.write(f'datadir: {args.datadir}\n')
            if model_time2:
                w.write(f'text2text model: {args.modelname}, \
                        batch_size: {args.batch_size}, epochs: {args.t2t_epoch}, lr: {args.t2t_lr}\n')
                w.write('text2text model train endtime: '+model_time2+'\n')
            if model_time3:
                w.write('pre_trn endtime: '+model_time3+'\n')
            if model_time4:
                w.write('pre_tst endtime: '+model_time4+'\n')
            if model_time5:
                w.write(f'combine model: {args.combine_model}, model_name: {args.combine_model_name}\n')
                w.write('combine endtime: '+model_time5+'\n')
            if model_time6:
                w.write(f'rank model: {args.rank_model}, rank_batch: {args.rank_batch}, \
                        rank_epoch: {args.rank_epoch} \n')
                w.write('rank model train endtime: '+model_time6+'\n')
                w.write(f'rank_save: {args.rankmodel_save} \n')
            if model_time7:
                w.write('ranking endtime: '+model_time7+'\n')
            if res:
                w.write(f'p@1:{res[0]:.6f} \nP@3:{res[1]:.6f}\nP@5:{res[2]:.6f}\n')
            w.write('end.'+'\n'+'\n')
            

# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--datadir', type=str, default='./dataset/EUR-Lex/',
#                         help='dataset_dir')
#     parser.add_argument('--test_json',type=str,default='test_finetune.json')
#     parser.add_argument('--train_json',type=str,default='train_finetune.json')
#     parser.add_argument('--all_labels',type=str,default='all_labels.txt')
#     parser.add_argument('--test_labels',type=str,default='test_labels.txt')
#     parser.add_argument('--train_labels',type=str,default="train_labels.txt")
#     parser.add_argument('--test_texts',type=str,default="test_texts.txt")
#     parser.add_argument('--train_texts',type=str,default="train_texts.txt")
#     # finetune args
#     parser.add_argument('--istrain',type=int,default=1,
#                         help="whether run finteune processing")
#     parser.add_argument('-b', '--batch_size', type=int, default=4,
#                         help='number of batch size for training')
#     parser.add_argument('-e', '--t2t_epoch', type=int, default=5,
#                         help='number of epochs to train (default: 100)')
#     parser.add_argument('--modelname', type=str,default='bart',
#                         help='modelname ')
#     parser.add_argument('--affix1',type=str,default="")
#     parser.add_argument('--affix2',type=str,default="")
#     parser.add_argument('--checkdir', type=str, default='bart_check',
#                         help='path to trained model to save')
#     parser.add_argument('--outputmodel',type=str,default='bart_save',
#                         help="fine-tune model save dir")
#     parser.add_argument('--t2t_lr', type=float, default=2e-5,
#                         help='learning rate')
#     parser.add_argument('--seed', type=int, default=44,
#                         help='random seed (default: 1)')
#     #perdicting args
#     parser.add_argument('--is_pred_trn',type=int,default=1,
#                         help="Whether run predicting training dataset")
#     parser.add_argument('--is_pred_tst',type=int,default=1,
#                         help="Whether run predicting testing dataset")
#     parser.add_argument('--top_k',type=int,default=10)
#     parser.add_argument('--data_size',type=int,default=12)
#     #combine part
#     parser.add_argument('--iscombine',type=int,default=1,
#                         help="Whether run combine")
#     parser.add_argument('--combine_model',type=str,default='bi-encoder')
#     parser.add_argument('--combine_model_name',type=str,default='all-MiniLM-L6-v2')
#     parser.add_argument('--combine_testdir',type=str,default="test_pred.txt")
#     parser.add_argument('--combine_traindir',type=str,default="train_pred.txt")
#     parser.add_argument('--combine_testout',type=str,default="test_combine_labels.txt")
#     parser.add_argument('--combine_trainout',type=str,default="train_combine_labels.txt")
#     #rank part
#     parser.add_argument('--is_rank_train',type=int,default=1)
#     parser.add_argument('--rank_model',type=str,default='all-MiniLM-L6-v2')
#     parser.add_argument('--rank_batch',type=int,default=64)
#     parser.add_argument('--rank_epoch',type=int,default=4)
#     parser.add_argument('--rankmodel_save',type=str,default='ba_bi_bi64')
#     parser.add_argument('--rank_textdir',type=str,default='train_texts.txt')
#     parser.add_argument('--is_ranking',type=int,default=1)
#     args = parser.parse_args()
#     run(args)
