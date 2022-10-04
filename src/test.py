from premethod import *

datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/']
# k_fold = [1,2,3,4,5]
tasks = ['test','train','valid']

# k_fold_split(datadir[0],datadir[0]+"K_fold/")
bart_clean(datadir[0]+"generate_result/"+tasks[0]+"_pred_kb.txt",datadir[0]+"generate_result/"+tasks[0]+"_pred_kb_c.txt")


        
            