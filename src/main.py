from argparse import ArgumentParser
from trainer_kp import *
from premethod import stem_labels,txt_to_json,get_all_labels,get_all_stemlabels,bart_clean
from pegasus_fine_tune import Pegasus_fine_tune
from generate_pegasus import *
from combine import *
from rank import rank, rank_bi
from rank_training import rank_train, rank_train_BI
from utils.p_at_1 import p_at_k
from keybart_finetune import *
from keybart_generate import get_pred_Keybart
from bart_finetune import fine_tune_bart
from bart_generate import get_pred_bart, get_pred_bart_batch
import  re
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
    
    # 注意文件路径
    datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/','./dataset/AmazonCat-13K/','./dataset/AmazonCat-13K-10/','./dataset/Wiki500K-20/']
    k_fold = [0,1,2,3,4]
    tasks = ['test','train','valid']
    models = {'pega':1,'bart':0,'kb':0}
    #datapreprocess(datadir[0])        
    gener = "res/"
    da =  1    
    if models['pega']:
        #fine_tune_pegasus_light(datadir[da],tasks[1]+'_finetune.json',tasks[1]+'_finetune.json',"pegasus_save_b","pegasus_check_b","google/bigbird-pegasus-large-bigpatent")
        #Pegasus_fine_tune(datadir[da],tasks[1]+'_finetune.json',tasks[1]+'_finetune.json',"pegasus_save","pegasus_check")
        for i in range(0):
            #get_pred_Pegasus_fast(datadir[da],gener+tasks[i]+"_pred0.txt",tasks[i]+"_finetune.json","pegasus_save")
            get_combine_bi_list(datadir[da],gener+tasks[i]+"_pred.txt","all_labels.txt",gener+tasks[i]+"_combine_labels_bi.txt")
        rank_train(datadir[da],'cross-encoder/stsb-roberta-base',tasks[1]+"_texts.txt",gener+tasks[1]+"_combine_labels_bi20.txt",tasks[1]+"_labels.txt","cr_en")
        rank(datadir[da],tasks[0]+"_texts.txt",gener+tasks[0]+"_combine_labels_bi.txt","cr_en",gener+tasks[0]+"_ranked_labels.txt")
        res = p_at_k(datadir[da],tasks[0]+"_labels.txt",gener+tasks[0]+"_combine_labels_bi.txt",datadir[da]+"res_pega.txt")
        res = p_at_k(datadir[da],tasks[0]+"_labels.txt",gener+tasks[0]+"_ranked_labels.txt",datadir[da]+"res_pega.txt")
    if models['bart']:
        #fine_tune_bart(datadir[da],tasks[1]+'_finetune.json',tasks[0]+'_finetune.json','bart_save','bart_check')
        for i in range(2):
            get_pred_bart_batch(datadir[da],gener+tasks[i]+"_pred_ba.txt",tasks[i]+"_finetune_a.json","bart_save")
            #bart_clean(datadir[1]+gener+tasks[i]+"_pred_ba.txt",datadir[1]+gener+tasks[i]+"_pred_ba_c.txt")
            #get_combine_bi_list(datadir[da],gener+tasks[i]+"_pred_ba_c.txt","all_labels.txt",gener+tasks[i]+"_combine_labels_ba_bi.txt")
        #rank_train_BI(datadir[da],tasks[1]+"_texts.txt",gener+tasks[1]+"_pred_ba.txt",tasks[1]+"_labels.txt","bi_en_ba")
        #rank_bi(datadir[da],tasks[0]+"_texts.txt",gener+tasks[0]+"_combine_labels_ba_bi.txt","bi_en_ba",gener+tasks[0]+"_ranked_labels_ba_bi.txt")
        #res = p_at_k(datadir[da],tasks[0]+"_labels.txt",gener+tasks[0]+"_pred_ba.txt",datadir[da]+"res_ba.txt")
        #res = p_at_k(datadir[da],tasks[0]+"_labels.txt",gener+tasks[0]+"_ranked_labels_ba_bi.txt",datadir[da]+"res_ba.txt")
    if models['kb']:    
        #kb_fine_tune(datadir[0],"kb_save","kb_check")
        fine_tune_keybart(datadir[0],tasks[1]+'_finetune.json',tasks[0]+'_finetune.json','keybart_save','keybart_test')
        for i in range(2):
            get_pred_Keybart(datadir[0],gener+tasks[i]+"_pred_kb.txt",tasks[i]+"_finetune.json","keybart_save")
            bart_clean(datadir[0]+gener+tasks[i]+"_pred_kb.txt",datadir[0]+gener+tasks[i]+"_pred_kb_c.txt")
            get_combine_list(datadir[0],gener+tasks[i]+"_pred_kb_c.txt","all_labels_sterm.txt",gener+tasks[i]+"_combine_labels_kb.txt")
    #get_combine_list(datadir[0],"generate_result/"+tasks[0]+"_pred_kb_c.txt","all_stemlabels.txt",tasks[0]+"_combine_labels_kb.txt")
    #get_combine_list(datadir[0],"generate_result/"+tasks[1]+"_pred_kb_c.txt","all_stemlabels.txt",tasks[1]+"_combine_labels_kb.txt")
        rank_train(datadir[0],tasks[1]+"_texts.txt",gener+tasks[1]+"_combine_labels_kb.txt",tasks[1]+"_labels_stem.txt","cr_en_kb")
        rank(datadir[0],tasks[0]+"_texts.txt",gener+tasks[0]+"_combine_labels_kb.txt","cr_en_kb",gener+tasks[0]+"_ranked_labels_kb.txt")
        res = p_at_k(datadir[0],tasks[0]+"_labels_stem.txt",gener+tasks[0]+"_ranked_labels_kb.txt",datadir[0]+"kb_res.txt")

# def datapreprocess(dir):
#     type = ['_labels', '_texts']
#     tasks = ['test', 'train']
#     #dir = './dataset/EUR-Lex/'
#     #词干化以及转化为json
#     dataset_path=[]
#     for i in range(len(tasks)):
#         label_path = dir+tasks[i]+type[0]
#         dataset_path.append(label_path+'_stem.txt')
#         stem_labels(label_path, label_path+"_stem")
#         text_path = dir+tasks[i]+type[1]
#         finetune_path = dir+tasks[i]+"_finetune"
#         txt_to_json(text_path, label_path+"_stem",
#                     finetune_path)  # 注意标签是已经stem过的
#     get_all_labels(dataset_path,dir+"all_labels.txt")
#     get_all_stemlabels(dir+'all_labels.txt',dir+'all_stemlabels.txt')
#     #stem_labels("./dataset/EUR-Lex/test_labels","./dataset/EUR-Lex/test_labels_stem")
#     #txt_to_json('./dataset/EUR-Lex/test_texts',"./dataset/EUR-Lex/test_labels_stem","./dataset/EUR-Lex/test_finetune")