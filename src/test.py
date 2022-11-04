from premethod import *
from cluster import *
datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/','./dataset/AmazonCat-13K/','./dataset/AmazonCat-13K-10/']
# k_fold = [1,2,3,4,5]
# labels =[]
# embeddings = get_embedding(datadir[1],"all_labels.txt")
# get_means(embeddings,datadir[1]+"all_label_cluster.txt")
#split_jsonfile(datadir[2]+"test_finetune.json",datadir[3]+"test_finetune.json",10)
#split_jsonfile(datadir[2]+"train_finetune.json",datadir[3]+"train_finetune.json",10)
json_to_text(datadir[3]+"test")
json_to_text(datadir[3]+"train")
get_all_labels_allfile([datadir[3]+"test_labels.txt",datadir[3]+"train_labels.txt"],datadir[3]+"all_labels.txt")
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
        
            