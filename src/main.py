from argparse import ArgumentParser
from train import *
from premethod import stem_labels,txt_to_json,get_all_labels,get_all_stemlabels,bart_clean
from pegasus_fine_tune import Pegasus_fine_tune
from generate_pegasus import *
from combine import *
from rank import rank, rank_bi
from rank_training import rank_train, rank_train_BI
from p_at_1 import p_at_k
from keybart_finetune import *
from keybart_generate import get_pred_Keybart
from bart_finetune import fine_tune_bart
from bart_generate import get_pred_bart, get_pred_bart_batch

def run(args:ArgumentParser):
    if args.istrain:
        #run finetune
        modeltrainer(args)
    if args.is_pre_trn:
        
        





if __name__ == '__main__':
    parser = ArgumentParser()
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
    parser.add_argument('--checkdir', type=str, default='model_check',
                        help='path to trained model to save')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=44,
                        help='random seed (default: 1)')
    parser.add_argument('--is_pred_trn',type=bool,default=True,
                        help="Whether run predicting training dataset")
    parser.add_argument('--is_pred_tst',type=bool,default=True,
                        help="Whether run predicting testing dataset")
    parser.add_argument()
    
    args = parser.parse_args()
    
    # 注意文件路径
    datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/','./dataset/AmazonCat-13K/','./dataset/AmazonCat-13K-10/','./dataset/Wiki500K-20/']
    k_fold = [0,1,2,3,4]
    tasks = ['test','train','valid']
    models = {'pega':0,'bart':1,'kb':0}
    #datapreprocess(datadir[0])
    if len(k_fold)==0:
        for j in k_fold:
            k_dir = "/K_fold/"+"K_"+str(j)+"/"
            if j==0:
                for i in range(2):
                    get_pred_Pegasus(datadir[0]+k_dir,tasks[i]+"_pred.txt",tasks[i]+".json","pegasus_save")
                    get_combine_list(datadir[0]+k_dir,tasks[i]+"_pred.txt","all_stemlabels.txt",tasks[i]+"_combine_labels.txt")
                rank_train(datadir[0]+k_dir,tasks[1]+"_texts.txt",tasks[1]+"_combine_labels.txt",tasks[1]+"_labels_stem.txt","cr_en_"+str(j))
                rank(datadir[0]+k_dir,tasks[0]+"_texts.txt",tasks[0]+"_combine_labels.txt","cr_en_"+str(j),tasks[0]+"_ranked_labels.txt")
                res = p_at_k(datadir[0]+k_dir,tasks[0]+"_labels_stem.txt",tasks[0]+"_ranked_labels.txt",datadir[0]+k_dir+"res.txt")
            else:   
                Pegasus_fine_tune(datadir[0]+k_dir,"pegasus_save","pegasus_check")
                for i in range(2):
                    get_pred_Pegasus(datadir[0]+k_dir,tasks[i]+"_pred.txt",tasks[i]+".json","pegasus_save")
                    get_combine_list(datadir[0]+k_dir,tasks[i]+"_pred.txt","all_stemlabels.txt",tasks[i]+"_combine_labels.txt")
                rank_train(datadir[0]+k_dir,tasks[1]+"_texts.txt",tasks[1]+"_combine_labels.txt",tasks[1]+"_labels_stem.txt","cr_en_"+str(j))
                rank(datadir[0]+k_dir,tasks[0]+"_texts.txt",tasks[0]+"_combine_labels.txt","cr_en_"+str(j),tasks[0]+"_ranked_labels.txt")
                res = p_at_k(datadir[0]+k_dir,tasks[0]+"_labels_stem.txt",tasks[0]+"_ranked_labels.txt",datadir[0]+k_dir+"res.txt")
        
    gener = "generate_result/"
    da =  1    
    if models['pega']:
        #fine_tune_pegasus_light(datadir[da],tasks[1]+'_finetune.json',tasks[1]+'_finetune.json',"pegasus_save_b","pegasus_check_b","google/bigbird-pegasus-large-bigpatent")
        #Pegasus_fine_tune(datadir[da],tasks[1]+'_finetune.json',tasks[1]+'_finetune.json',"pegasus_save","pegasus_check")
        for i in range(0):
            get_pred_Pegasus_fast(datadir[da],gener+tasks[i]+"_pred.txt",tasks[i]+"_finetune.json","pegasus_save_b")
            get_combine_bi_list(datadir[da],gener+tasks[i]+"_pred.txt","all_labels.txt",gener+tasks[i]+"_combine_labels_bi.txt")
        rank_train_BI(datadir[da],tasks[1]+"_texts.txt",gener+tasks[1]+"_combine_labels_bi.txt",tasks[1]+"_labels.txt","bi_en")
        rank_bi(datadir[da],tasks[0]+"_texts.txt",gener+tasks[0]+"_combine_labels_bi.txt","bi_en",gener+tasks[0]+"_ranked_labels_bi.txt")
        res = p_at_k(datadir[da],tasks[0]+"_labels.txt",gener+tasks[0]+"_combine_labels_bi.txt",datadir[da]+"res_pega.txt")
        res = p_at_k(datadir[da],tasks[0]+"_labels.txt",gener+tasks[0]+"_ranked_labels_bi.txt",datadir[da]+"res_pega.txt")
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