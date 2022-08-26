
from premethod import *
from pegasus_fine_tune import *
#from head import*
from generate_pegasus import *
from combine import *
from rank_training import rank_train
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


def fine_tune(dir,task):
    #dir = './dataset/EUR-Lex/'
    # use XSum dataset as example, with first 1000 docs as training data
    prefix = "summarize: "
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    from datasets import load_dataset
    # dataset = load_dataset("xsum")
    dataset = load_dataset('json',data_files={'train': dir+'train_finetune.json', 'valid': dir+'test_finetune.json'}).shuffle(seed=42)
    train_texts, train_labels = [prefix + each for each in dataset['train']['document']], dataset['train']['summary']
    valid_texts, valid_labels = [prefix + each for each in dataset['valid']['document']], dataset['valid']['summary']
    # use Pegasus Large model as base for fine-tuning
    model_name = 'google/pegasus-large'
    #return train_dataset, val_dataset, test_dataset, tokenizer 可以一起投入
    train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
    valid_dataset, _, _, _ = prepare_data(model_name, valid_texts, valid_labels)
    trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=valid_dataset)
    #,val_dataset=valid_dataset
    print("start training")
    start_time = time.time()
    trainer.train()
    trainer.save_model(output_dir=dir+'pegasus_save')
    end_time = time.time()
    print('pegasus_time_cost: ',end_time-start_time,'s')


if __name__ == '__main__':
    # 注意文件路径
    datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/']
    tasks = ['test','train','valid']
    #datapreprocess(datadir[0])
    #fine_tune(datadir[0])
    #get_pred_Pegasus(datadir[0],datadir[0]+"generate_result/"+tasks[1]+"_pred.txt",datadir[0]+tasks[1]+"_finetune.json","pegasus_save")
    #get_combine_list(datadir[0],"generate_result/"+tasks[1]+"_pred.txt","all_stemlabels.txt",tasks[1]+"_combine_label.txt")
    rank_train(datadir[0],"train_texts.txt","generate_result/train_pred.txt","train_combine_label.txt","cr_en")