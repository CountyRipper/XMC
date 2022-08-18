from premethod import *
from pegasus_fine_tune import *
#from head import*


def datapreprocess():
    type = ['_labels', '_texts']
    tasks = ['test', 'train']
    dir = './dataset/EUR-Lex/'
    for i in range(len(tasks)):
        label_path = dir+tasks[i]+type[0]
        stem_labels(label_path, label_path+"_stem")
        text_path = dir+tasks[i]+type[1]
        finetune_path = dir+tasks[i]+"_finetune"
        txt_to_json(text_path, label_path+"_stem",
                    finetune_path)  # 注意标签是已经stem过的
    # stem_labels("./dataset/EUR-Lex/test_labels","./dataset/EUR-Lex/test_labels_stem")
    # txt_to_json('./dataset/EUR-Lex/test_texts',"./dataset/EUR-Lex/test_labels_stem","./dataset/EUR-Lex/test_finetune")


def fine_tune():
    dir = './dataset/EUR-Lex/'
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
    trainer.save_model(output_dir=dir+'pegasus_test_save')
    end_time = time.time()
    print('pegasus_time_cost: ',end_time-start_time,'s')


if __name__ == '__main__':
    # 注意文件路径
    datapreprocess()
    fine_tune()
