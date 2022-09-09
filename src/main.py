from premethod import *
from pegasus_fine_tune import *
from pegasus_fine_tune1 import *
from generate_pegasus import *
from combine import *
from rank_training import rank_train
from rank import rank
from p_at_1 import p_at_k
from keybart_finetune import *
from keybart_generate import get_pred_Keybart
from bart_finetune import fine_tune_bart
from bart_generate import get_pred_bart
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
    models = {'pega':1,'bart':0,'kb':0}
    #datapreprocess(datadir[0])
    if models['pega']:
        #Pegasus_fine_tune(datadir[0],"pegasus_xs_save","pegasus_xs_check")
        for i in range(2):
            get_pred_Pegasus_fast(datadir[0],"generate_result/"+tasks[i]+"_pred_xs.txt",tasks[i]+"_finetune.json","pegasus_save")
            get_combine_list(datadir[0],"generate_result/"+tasks[i]+"_pred_xs.txt","all_stemlabels.txt",tasks[i]+"_combine_labels_xs.txt")
    if models['bart']:
        fine_tune_bart(datadir[0],tasks[1]+'_finetune.json',tasks[0]+'_finetune.json','bart_save','bart_check')
        for i in range(2):
            get_pred_bart(datadir[0],"generate_result/"+tasks[i]+"_pred_ba.txt",tasks[i]+"_finetune.json","bart_save")
            get_combine_list(datadir[0],"generate_result/"+tasks[i]+"_pred_ba.txt","all_stemlabels.txt",tasks[i]+"_combine_labels_ba.txt")
    else:    
        #kb_fine_tune(datadir[0],"kb_save","kb_check")
        fine_tune_keybart(datadir[0],tasks[1]+'_finetune.json',tasks[0]+'_finetune.json','keybart_save','keybart_test')
        get_pred_Keybart(datadir[0],"generate_result/"+tasks[0]+"_kb_pred.txt",tasks[0]+"_finetune.json","keybart_save")
        get_pred_Keybart(datadir[0],"generate_result/"+tasks[1]+"_kb_pred.txt",tasks[1]+"_finetune.json","keybart_save") 
    #get_combine_list(datadir[0],"generate_result/"+tasks[0]+"_pred.txt","all_stemlabels.txt",tasks[0]+"_combine_labels.txt")
    #get_combine_list(datadir[0],"generate_result/"+tasks[1]+"_pred.txt","all_stemlabels.txt",tasks[1]+"_combine_labels.txt")
    #rank_train(datadir[0],tasks[1]+"_texts.txt",tasks[1]+"_combine_labels1.txt",tasks[1]+"_labels_stem.txt","cr_en")
    #rank(datadir[0],tasks[0]+"_texts.txt",tasks[0]+"_combine_labels1.txt","cr_en",tasks[0]+"_ranked_labels1.txt")
    #res = p_at_k(datadir[0],tasks[0]+"_labels_stem.txt",tasks[0]+"_ranked_labels1.txt")
    