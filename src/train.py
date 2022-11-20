import json
import os
import torch
import tqdm
from transformers import (AutoTokenizer,BartTokenizer, BartForConditionalGeneration,Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments,PegasusForConditionalGeneration, PegasusTokenizer,
                          T5ForConditionalGeneration,T5Tokenizer,AutoModelForSeq2SeqLM)
import time,datetime
from torch.utils.data import DataLoader
class MyData(torch.utils.data.Dataset):
    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels= labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

class modeltrainer(object):
    def __init__(self,args) -> None:
        self.datadir = args.datadir
        self.modelname = args.modelname
        self.checkdir = self.datadir +args.checkdir
        self.output = self.datadir + args.output
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.modelname=='bart-large' or self.modelname=='BART-large' or self.modelname=='Bart-large':
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large",cache_dir='./models').to(self.device)
            self.tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large",cache_dir='./models')
        elif self.modelname=='bart' or self.modelname=='BART' or self.modelname=='Bart':
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base",cache_dir='./models').to(self.device)
            self.tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-base",cache_dir='./models')
        elif self.modelname=='pegasus' or self.modelname=='Pegasus':
            self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large',cache_dir='./models').to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large",cache_dir='./models')
        elif self.modelname=='pegasus-xsum' or self.modelname=='Pegasus-xsum':
            self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum',cache_dir='./models').to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large",cache_dir='./models')
        elif self.modelname=='t5' or self.modelname=='T5':
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base",cache_dir='./models').to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base",cache_dir='./models')
        elif self.modelname=='t5-large' or self.modelname=='T5-large':
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large",cache_dir='./models').to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large",cache_dir='./models')
        elif self.modelname=='keybart' or  self.modelname=='KeyBART':
            self.tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART",cache_dir='./models')
            self.model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART",cache_dir='./models').to(self.device)
        #self.myData = MyData
    def __token_data(self,texts,labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        decodings = self.tokenizer(labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        return dataset_tokenized
    def __finetune(self,freeze_encoder=None):
        train_dir= self.datadir+"train_finetune.json"
        valid_dir= self.datadir+"test_finetune.json"
        print('modelname:'+self.modelname)
        print('checkdir:'+self.checkdir)
        print('save_dir:'+self.output)
        print('batch_size:'+self.batch_size)
        print('epoch:'+self.epoch)       
        from datasets import load_dataset
        prefix = "summarize: "
        dataset = load_dataset('json',data_files={'train': train_dir, 'valid': valid_dir}).shuffle(seed=42)
        train_texts, train_labels = [prefix + each for each in dataset['train']['document']], dataset['train']['summary']
        valid_texts, valid_labels = [prefix + each for each in dataset['valid']['document']], dataset['valid']['summary']
        train_dataset = self.__token_data(train_texts,train_labels)
        valid_dataset = self.__token_data(valid_texts,valid_labels)
        if freeze_encoder:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
        train_args = Seq2SeqTrainingArguments(
            output_dir=self.checkdir,
            num_train_epochs=self.epoch,           # total number of training epochs
            per_device_train_batch_size=self.batch_size,   # batch size per device during training, can increase if memory allows
            per_device_eval_batch_size=self.batch_size,    # batch size for evaluation, can increase if memory allows
            save_steps=30000,                  # number of updates steps before checkpoint saves
            save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
            evaluation_strategy = "epoch",     # evaluation strategy to adopt during training                 # number of update steps before evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            predict_with_generate=True,
        )
        self.trainer = Seq2SeqTrainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=train_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,            # evaluation dataset
            tokenizer=self.tokenizer
        )
        self.trainer.train()
        self.trainer.save_model(self.output)
        
    def train(self):
        start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('train start:'+start)
        time_stap1 = time.clock()
        self.__finetune()
        time_stap2 = time.clock()
        end =  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('tarin end:'+ end)
        print('tarining cost time:'+ str((time_stap2-time_stap1)/60/60 )+"hours.")
        with open(self.datadir+"train_log.txt",'a+')as w: 
            w.write("datadir:"+self.datadir+", "+"model_name: "+self.modelname+", "+"batch_size: "+self.batch_size+"epoch: "+self.epoch+"\n"
                    "checkdir: "+self.checkdir+", "+"output: "+self.output+"\n")
            w.write("starttime:"+start+". ")
            w.write("endtime: "+end+"\n")
    
    def __predict(self,model,tokenizer,documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(self.device)
        #inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
      # Generate Summary
        summary_ids = self.model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 5).to(self.device)
        #summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
        pre_result=self.tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2)
        #pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])  #[2:-2]
        return pre_result   
    
    def predicting(self,src_dataname,output_dir):
        output_dir = self.datadir+output_dir
        print('output: '+output_dir)
        src_dataname = self.datadir+src_dataname
        print('src_data: '+src_dataname)
        self.tokenizer.save_pretrained(self.modelname+"_tokenizer")
        self.tokenizer.save_vocabulary(self.modelname+"_tokenizer")
        data = []
        dic = [] # dictionary for save each model generate result
        src_value = [] # using for get source document which is used to feed into model, and get predicting result
        res = []
        batch=[]
        # open test file 
        with open(src_dataname, 'r+') as f:
            for line in f:
                data.append(json.loads(line)['document'])
            # 进度条可视化 vision process
            dataloader = DataLoader(data,batch_size=32)
            f=open(output_dir,'w+')
            f.close()
            with open(output_dir,'a+') as t:
                for i in tqdm(dataloader): #range(len(data))
                    batch = i
                    tmp_result = self.__predict(self.model,self.tokenizer,batch)
                    for j in tmp_result:
                        l_labels = [] #l_label 是str转 label的集合
                        pre = j.strip('[]').strip().split(",")
                        for k in range(len(pre)):
                            tmpstr = pre[k].strip(" ").strip("'").strip('"')
                            if tmpstr=='':continue
                            l_labels.append(tmpstr)
                        res.append(l_labels)
                        t.write(", ".join(l_labels))
                        t.write("\n")
        return res 
        
           
    