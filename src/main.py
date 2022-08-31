from premethod import *
from pegasus_fine_tune import *
#from head import*
from generate_pegasus import *
from combine import *
from rank_training import rank_train
from rank import rank
from p_at_1 import p_at_k
from keybart_finetune import *
def datapreprocess(dir):
    type = ['_labels', '_texts']
    tasks = ['test', 'train']
    #dir = './dataset/EUR-Lex/'
    #词干化以及转化为json
    dataset_path=[]
    for i in range(len(tasks)):
        label_path = dir+tasks[i]+type[0]
        dataset_path.append(label_path+'_stem.txt')
        stem_labels(label_path, label_path+"_stem")
        text_path = dir+tasks[i]+type[1]
        finetune_path = dir+tasks[i]+"_finetune"
        txt_to_json(text_path, label_path+"_stem",
                    finetune_path)  # 注意标签是已经stem过的
    get_all_labels(dataset_path,dir+"all_labels.txt")
    get_all_stemlabels(dir+'all_labels.txt',dir+'all_stemlabels.txt')
    #stem_labels("./dataset/EUR-Lex/test_labels","./dataset/EUR-Lex/test_labels_stem")
    #txt_to_json('./dataset/EUR-Lex/test_texts',"./dataset/EUR-Lex/test_labels_stem","./dataset/EUR-Lex/test_finetune")




if __name__ == '__main__':
    # 注意文件路径
    datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/']
    tasks = ['test','train','valid']
    #datapreprocess(datadir[0])
    #Pegasus_fine_tune(datadir[0])
    fine_tune_keybart(datadir[0],tasks[1]+'_finetune.json',tasks[0]+'_finetune.json','keybart_save')
    #get_pred_Pegasus(datadir[0],datadir[0]+"generate_result/"+tasks[0]+"_pred.txt",datadir[0]+tasks[0]+"_finetune.json","pegasus_save")
    #get_combine_list(datadir[0],"generate_result/"+tasks[0]+"_pred.txt","all_stemlabels.txt",tasks[0]+"_combine_labels.txt")
    #get_combine_list(datadir[0],"generate_result/"+tasks[1]+"_pred.txt","all_stemlabels.txt",tasks[1]+"_combine_labels.txt")
    #rank_train(datadir[0],tasks[1]+"_texts.txt",tasks[1]+"_combine_labels.txt",tasks[1]+"_labels_stem.txt","cr_en")
    #rank(datadir[0],tasks[0]+"_texts.txt",tasks[0]+"_combine_labels.txt","cr_en",tasks[0]+"_ranked_labels.txt")
    #res = p_at_k(datadir[0],tasks[0]+"_labels_stem.txt",tasks[0]+"_ranked_labels.txt")
    