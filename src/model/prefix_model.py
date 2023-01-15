import os
from typing import Any
import pytorch_lightning as pl

import torch
from unified.prefixtuning import PrefixModel
from transformers import (AutoTokenizer,BartTokenizerFast, BartForConditionalGeneration,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,PegasusForConditionalGeneration, PegasusTokenizerFast,
                          T5ForConditionalGeneration,T5TokenizerFast,AutoModelForSeq2SeqLM)

from torch.utils.data import DataLoader
import numpy as np
import yaml
from datasets import load_dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class MyData(torch.utils.data.Dataset):
    def __init__(self, encoding, labels):
        self.ids = encoding['input_ids']
        self.mask = encoding['attention_mask']
        self.labels= labels['input_ids']
    def __getitem__(self, idx):
      item={}
      item['input_ids'] = torch.tensor(self.ids[idx]).to(device)
      item['attention_mask'] = torch.tensor(self.mask[idx]).to(device)
      item['labels'] = torch.tensor(self.labels[idx]).to(device)
      #item={'input_ids': torch.tensor(val[idx]).to(device) for key, val in self.encoding.items()}
      #item['labels'] = torch.tensor(self.labels['input_ids'][idx]).to(device)
      return item
    def __len__(self):
        return len(self.labels)  # len(self.labels)

class prefix_args(object):
    def __init__(self) -> None:
        self.prefix_sequence_length=100
        self.prefix_mid_dim=512
        self.prefix_dropout=0.0
        self.bert_location="facebook/bart-base"
        self.model_knowledge_usage="concatenate"
        self.model_freeze_plm=0
        self.model_freeze_prefix=0
        self.model_map_description=0
        self.model_use_description=0
        self.special_tokens=None
    
class Bart_Prefix_Model(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.prefix_hparameters = prefix_args()
        self.hparameters = hparams
        self.model = PrefixModel(self.prefix_hparameters).to(device)
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.save_hyperparameters()
    def get_tokenizer(model_name):
        if 'bart' in model_name:
            return BartTokenizerFast.from_pretrained(model_name)
        if 't5' in model_name:
            return T5TokenizerFast.from_pretrained(model_name)
        
                
    def forward(self, encoder_input_ids,attention_mask,decoder_input_ids):
        return self.model(encoder_input_ids,attention_mask,decoder_input_ids)  
    def training_step(self, batch, batch_idx):
        encoder_input_ids, encoder_attention_mask,labels = torch.stack([i['input_ids'] for i in batch ]),torch.stack([i['attention_mask'] for i in batch ]),torch.stack([i['labels'] for i in batch ])
        res = self(encoder_input_ids,encoder_attention_mask,labels)
        #loss = self.custom_loss(logits, labels) custom loss
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # #手动优化scheduler
        sch = self.lr_schedulers()
        if (batch_idx + 1) % 10 == 0:
            sch.step()
        self.log('lr',cur_lr, prog_bar=True, on_step=True)
        self.log('train_loss', res['loss'], prog_bar=True,batch_size=self.hparameters['batch_size'])
        return res['loss']   
    def validation_step(self, batch, batch_idx):
        encoder_input_ids, encoder_attention_mask,labels = torch.stack([i['input_ids'] for i in batch ]),torch.stack([i['attention_mask'] for i in batch ]),torch.stack([i['labels'] for i in batch ])
        #encoder_input_ids, encoder_attention_mask,labels = batch['input_ids'],batch['attention_mask'],batch['labels']
        res = self(encoder_input_ids,encoder_attention_mask,labels)
        # cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # self.log('lr',cur_lr, prog_bar=True, on_step=True)
        #loss, logits = self(encoder_input_ids,labels)    
        self.log('val_loss', res['loss'], prog_bar=True,batch_size=self.hparameters['batch_size'])
        return res['loss']
    def generate(self, input_ids,attention_mask,**kwargs) -> Any:
        return self.model.generate(input_ids,attention_mask,**kwargs)     
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparameters['learning_rate'])
        # self.lr_scheduler = CosineWarmupScheduler(
        #     optimizer, warmup=self.hparameters['warmup'], max_iters=self.hparameters['max_iters']
        # )
        # return optimizer
        return{
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100),
            # "lr_scheduler":torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.99),
            "interval": "step",
            "frequency": 1,
        }
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.StepLR(
        #         optimizer,
        #         step_size=50,
        #         gamma=0.1
        #     ),
        #     "interval": "step",
        #     "frequency": 20,
        # }
        # #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=50,gamma=0.1)
        # return [optimizer],[scheduler]
      
    def train_dataloader(self):
        datadir = self.hparameters['train_dir']
        #prefix = "summarize: "
        dataset = load_dataset('json',data_files= datadir)
        train_texts, train_labels = [ each for each in dataset['train']['document']], dataset['train']['summary']
        
        encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        decodings = self.tokenizer(train_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        train_data = DataLoader(dataset_tokenized,batch_size= self.hparameters['batch_size'],collate_fn=lambda x: x,shuffle=True)
        # create a dataloader for your training data here
        return train_data 
    def val_dataloader(self):
        datadir = self.hparameters['val_dir']
        #prefix = "summarize: "
        dataset = load_dataset('json',data_files=datadir)
        val_texts, val_labels = [ each for each in dataset['train']['document']], dataset['train']['summary']

        encodings = self.tokenizer(val_texts, truncation=True, padding=True)
        decodings = self.tokenizer(val_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        print(len(dataset_tokenized))
        val_data = DataLoader(dataset_tokenized,batch_size= self.hparameters['batch_size'],collate_fn=lambda x: x,shuffle=True)
        # create a dataloader for your training data here
        return val_data



