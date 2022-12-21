#from premethod import *
#from cluster import *
# import json
# import re
# from utils.premethod import get_all_stemlabels, label_index_to_label_raw, split_jsonfile, split_txt,txt_to_json
# from tqdm import tqdm
# datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/','./dataset/AmazonCat-13K/','./dataset/AmazonCat-13K-10/','./dataset/Wiki10-31K/','./dataset/Wiki500K-10/']
# # k_fold = [1,2,3,4,5]
# labels_list=[]
# js=[]
import torch
from model.rank_model import Rank_model,rankdata
from torch.utils.data import DataLoader
from utils.premethod import read_labels,read_texts
import pytorch_lightning as pl
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--warmup_epochs',type=int,default=2)
parser.add_argument('--update_step',type=int,default=200)
parser.add_argument('--bert_name',type=str,default='bert-uncased-base')
parser.add_argument('--feature_layers',type=int,default=5)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--hidden_dim',type=int,default=300)
parser.add_argument('--labels_num',type=int,default=30522)
parser.add_argument('--batch_size',type=int,default=6)
parser.add_argument('--max_epochs',type=int,default=6)
parser.add_argument('--num_training_samples',type=int,default=0)
parser.add_argument('--learning_rate',type=float,default=1e-5)

args = parser.parse_args()
rank_model = Rank_model.load_from_checkpoint("./log/rank_check/epoch=8-val_loss=0.68-other_metric=0.00.ckpt",
                                             args)
texts= read_texts("./dataset/EUR-Lex/test_texts.txt")
pre_labels = []
test_combine_labels = read_labels("./dataset/EUR-Lex/res/test_combine_labels_t5lbi.txt")

test_data = rankdata(texts,pre_labels)
test_dataloder = DataLoader(test_data,batch_size=8,collate_fn=lambda x: x,shuffle=True)
trainer = pl.Trainer()
res = trainer.predict(rank_model,test_dataloder)
# #split_jsonfile(datadir[1]+"test_finetune.json",datadir[1]+"test_finetune10.json")
# #split_jsonfile(datadir[1]+"train_finetune.json",datadir[1]+"train_finetune10.json")
# # split_txt(datadir[1]+"train_labels.txt",datadir[1]+"train_labels10.txt")    
# # split_txt(datadir[1]+"train_texts.txt",datadir[1]+"train_texts10.txt")  
# # split_txt(datadir[1]+"test_labels.txt",datadir[1]+"test_labels10.txt")  
# # split_txt(datadir[1]+"test_texts.txt",datadir[1]+"test_texts10.txt")  
# get_all_stemlabels(datadir[5]+"all_labels.txt",datadir[5]+"train_labels.txt",datadir[5]+"test_labels.txt")    
# with open(datadir[1]+"res/test_combine_labels_pegabi1.txt",'w+') as w:
#     for i in labels_list:
#         w.write(", ".join(i))
#         w.write('\n')
    
#label_index_to_label_raw(datadir[4]+'test_labels_ind.txt',datadir[4]+'all_labels.txt',datadir[4]+'test_labels.txt')
#label_index_to_label_raw(datadir[4]+'train_labels_ind.txt',datadir[4]+'all_labels.txt',datadir[4]+'train_labels.txt')
# txt_to_json(datadir[4]+'test_texts',datadir[4]+'test_labels',datadir[4]+'test_finetune')
# txt_to_json(datadir[4]+'train_texts',datadir[4]+'train_labels',datadir[4]+'train_finetune')
# label_list = []
# with open(datadir[1]+'all_labels.txt','r') as r:
#     for i in tqdm(r):
#         label_list.append(i.rstrip('\n'))
# label_list = list(map(lambda x: x.replace('_',' '),label_list))
# with open(datadir[1]+'all_labels1.txt','w+') as w:
#     for i in tqdm(label_list):
#         w.write(i+'\n')


# labels =[]
# embeddings = get_embedding(datadir[1],"all_labels.txt")
# get_means(embeddings,datadir[1]+"all_label_cluster.txt")
#split_jsonfile(datadir[2]+"test_finetune.json",datadir[3]+"test_finetune.json",10)
#split_jsonfile(datadir[2]+"train_finetune.json",datadir[3]+"train_finetune.json",10)
#json_to_text(datadir[3]+"test")
#json_to_text(datadir[3]+"train")
#get_all_labels_allfile([datadir[3]+"test_labels.txt",datadir[3]+"train_labels.txt"],datadir[3]+"all_labels.txt")
# with open('./dataset/Wiki500K/Yf.txt','r+') as f:
#     for row in f:
#         labels.append(row.replace("Category:","").split("->")[0])
# with open('./dataset/Wiki500K/all_labels.txt','w+') as w:
    
#     for i in labels:
#         w.write(i+"\n")
# print("./dataset/Wiki500K/all_labels.txt")
#tasks = ['test','train','valid']
#amazoncat_change(datadir[2]+"test.json",datadir[2]+"Yf.txt" ,datadir[2]+"test_finetune.json",datadir[2]+"test_labels.txt",datadir[2]+"test_texts.txt")
#amazoncat_change(datadir[2]+"train.json",datadir[2]+"Yf.txt" ,datadir[2]+"train_finetune.json",datadir[2]+"train_labels.txt",datadir[2]+"train_texts.txt")

# k_fold_split(datadir[0],datadir[0]+"K_fold/")
#bart_clean(datadir[0]+"generate_result/"+tasks[0]+"_pred_kb.txt",datadir[0]+"generate_result/"+tasks[0]+"_pred_kb_c.txt")
#split_jsonfile(datadir[1]+"trn.json",datadir[1]+"train_split.json")
# res = set()
# with open(datadir[0]+"train_labels.txt", 'r+') as f:
#     for row in f:
#         for i in row.strip("\n").split(" "):
#             res.add(i)
# with open(datadir[0]+"test_labels.txt", 'r+') as f:
#     for row in f:
#         for i in row.strip("\n").split(" "):
#             res.add(i)
# res = list(res)
# res.sort()

# with open(datadir[0]+"label_sort_raw.txt","w+") as w:
#     for i in res:
#         w.write(i+"\n") 
#split_jsonfile(datadir[1]+"train_finetune.json",datadir[1]+"train_finetune_s50.json",5)
#wiki500_change(datadir[1]+"test_split.json",datadir[1]+"Yf.txt" ,datadir[1]+"test_finetune.json",datadir[1]+"test_labels.txt",datadir[1]+"test_texts.txt")
#wiki500_change(datadir[1]+"train_split.json",datadir[1]+"Yf.txt" ,datadir[1]+"train_finetune.json",datadir[1]+"train_labels.txt",datadir[1]+"train_texts.txt")
        
            