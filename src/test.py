#from premethod import *
#from cluster import *
import re
from utils.premethod import label_index_to_label_raw,txt_to_json
from tqdm import tqdm
datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/','./dataset/AmazonCat-13K/','./dataset/AmazonCat-13K-10/','./dataset/Wiki10-31K/']
# k_fold = [1,2,3,4,5]
labels_list=[]
with open(datadir[1]+"res/test_combine_labels_pegabi.txt",'r+') as r:
    for i in r:
        labels_list.append(i.rstrip('\n').split(', '))
count=0
for i in range(len(labels_list)):

    tmp = []
    for j in labels_list[i]:
        if j not in tmp:
            tmp.append(j)
        else:
            count=count+1
    labels_list[i] = list(tmp)
print('count',count)
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
        
            