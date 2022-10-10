from premethod import *

datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/']
# k_fold = [1,2,3,4,5]
tasks = ['test','train','valid']

# k_fold_split(datadir[0],datadir[0]+"K_fold/")
#bart_clean(datadir[0]+"generate_result/"+tasks[0]+"_pred_kb.txt",datadir[0]+"generate_result/"+tasks[0]+"_pred_kb_c.txt")
#split_jsonfile(datadir[1]+"trn.json",datadir[1]+"train_split.json")
split_jsonfile(datadir[1]+"train_finetune.json",datadir[1]+"train_finetune_s100.json")
wiki500_change(datadir[1]+"train_split.json",datadir[1]+"Yf.txt" ,datadir[1]+"train_finetune.json",datadir[1]+"train_label.txt",datadir[1]+"train_texts.txt")

        
            