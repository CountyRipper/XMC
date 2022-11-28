
from combine import get_combine_list
from utils.generate_pegasus import get_pred_Pegasus
from utils.pegasus_fine_tune import Pegasus_fine_tune
from rank import rank
from rank_training import rank_train
from utils.p_at_1 import p_at_k


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