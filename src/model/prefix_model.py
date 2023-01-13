import os
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
import torch
from unified.prefixtuning import PrefixModel
from transformers import (AutoTokenizer,BartTokenizerFast, BartForConditionalGeneration,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,PegasusForConditionalGeneration, PegasusTokenizerFast,
                          T5ForConditionalGeneration,T5TokenizerFast,AutoModelForSeq2SeqLM)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from datasets import load_dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        self.prefix_mid_dim=1024
        self.prefix_dropout=0.1
        self.bert_location="facebook/bart-base"
        self.model_knowledge_usage="concatenate"
        self.model_freeze_plm=1
        self.model_freeze_prefix=0
        self.model_map_description=0
        self.model_use_description=0
        self.special_tokens=None
    
class Bart_Prefix_Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.prefix_hparameters = prefix_args()
        self.hparameters = hparams
        self.model = PrefixModel(self.prefix_hparameters)
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
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
        self.log('train_loss', res['loss'], prog_bar=True,batch_size=self.hparameters['batch_size'])
        return res['loss']   
    def validation_step(self, batch, batch_idx):
        encoder_input_ids, encoder_attention_mask,labels = torch.stack([i['input_ids'] for i in batch ]),torch.stack([i['attention_mask'] for i in batch ]),torch.stack([i['labels'] for i in batch ])
        #encoder_input_ids, encoder_attention_mask,labels = batch['input_ids'],batch['attention_mask'],batch['labels']
        res = self(encoder_input_ids,encoder_attention_mask,labels)
        #loss, logits = self(encoder_input_ids,labels)    
        self.log('val_loss', res['loss'], prog_bar=True,batch_size=self.hparameters['batch_size'])
        return res['loss']
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)     
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparameters['learning_rate'])  
    def train_dataloader(self):
        datadir = self.hparameters['train_dir']
        prefix = "summarize: "
        dataset = load_dataset('json',data_files= datadir)
        train_texts, train_labels = [prefix + each for each in dataset['train']['document']], dataset['train']['summary']

        encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        decodings = self.tokenizer(train_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        train_data = DataLoader(dataset_tokenized,batch_size= self.hparameters['batch_size'],collate_fn=lambda x: x,shuffle=True)
        # create a dataloader for your training data here
        return train_data 
    def val_dataloader(self):
        datadir = self.hparameters['val_dir']
        prefix = "summarize: "
        dataset = load_dataset('json',data_files=datadir)
        val_texts, val_labels = [prefix + each for each in dataset['train']['document']], dataset['train']['summary']

        encodings = self.tokenizer(val_texts, truncation=True, padding=True)
        decodings = self.tokenizer(val_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        print(len(dataset_tokenized))
        val_data = DataLoader(dataset_tokenized,batch_size= self.hparameters['batch_size'],collate_fn=lambda x: x,shuffle=True)
        # create a dataloader for your training data here
        return val_data

hparams = {
    'max_epochs': 4,
    'batch_size': 4,
    'learning_rate': 5e-5,
    'train_dir': "./dataset/Wiki10-31K/train_finetune.json",
    'val_dir': "./dataset/Wiki10-31K/test_finetune.json",
    'data_dir': "./dataset/Wiki10-31K/",
    'model': 'BART'
}

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
checkpoint_callback = ModelCheckpoint(
        dirpath='./log/prefix_check',
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
    )
lr_callback = LearningRateMonitor(logging_interval="step")
model = Bart_Prefix_Model(hparams)
logger = TensorBoardLogger(save_dir=os.path.join(hparams['data_dir'],'prefix'),name=hparams['model']+'_log')

trainer = pl.Trainer(max_epochs=3, callbacks=[early_stopping], logger=logger,
                     default_root_dir=os.path.join(hparams['data_dir'],hparams['model']+'_save'),
                     enable_checkpointing=True,
                     
                     #auto_lr_find=True,
                     accelerator="gpu", devices=1)
trainer.fit(model,train_dataloaders=model.train_dataloader(),
                val_dataloaders=model.val_dataloader())