import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
import torch
from transformers import (AutoTokenizer,BartTokenizerFast, BartForConditionalGeneration,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,PegasusForConditionalGeneration, PegasusTokenizerFast,
                          T5ForConditionalGeneration,T5TokenizerFast,AutoModelForSeq2SeqLM)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
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

class GenerationModel(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparameters = hparams
    if ('bart' in hparams['model']) or ('Bart'in hparams['model'])or ('BART'in hparams['model']):
      if ('bart-large'in hparams['model']) or ('Bart-large'in hparams['model']) or ('BART-large'in hparams['model']):
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
      else:
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    if ('pegasus'in hparams['model']) or ('Pegasus'in hparams['model']):
      self.tokenizer = PegasusTokenizerFast.from_pretrained('google/pegasus-large')
      self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large').to(device)
    if ('t5'  in hparams['model']) or ( 'T5'in hparams['model']):
      if ('t5-large'in hparams['model']) or ('T5-large'in hparams['model']):
        self.tokenizer = T5TokenizerFast.from_pretrained('google/t5-v1_1-large')
        self.model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-large').to(device)
      else:
        self.tokenizer = T5TokenizerFast.from_pretrained('google/t5-v1_1-base')
        self.model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base').to(device)
    
          
  def forward(self, encoder_input_ids,decoder_input_ids):
    return self.model(input_ids=encoder_input_ids, labels=decoder_input_ids)

  def training_step(self, batch, batch_idx):
    encoder_input_ids, encoder_attention_mask,labels = torch.stack([i['input_ids'] for i in batch ]),torch.stack([i['attention_mask'] for i in batch ]),torch.stack([i['labels'] for i in batch ])
    res = self(encoder_input_ids,labels)
    #loss = self.custom_loss(logits, labels) custom loss
    self.log('train_loss', res.loss, prog_bar=True,batch_size=self.hparameters['batch_size'])
    return res.loss

  def validation_step(self, batch, batch_idx):
    encoder_input_ids, encoder_attention_mask,labels = torch.stack([i['input_ids'] for i in batch ]),torch.stack([i['attention_mask'] for i in batch ]),torch.stack([i['labels'] for i in batch ])
    #encoder_input_ids, encoder_attention_mask,labels = batch['input_ids'],batch['attention_mask'],batch['labels']
    res = self(encoder_input_ids,labels)
    #loss, logits = self(encoder_input_ids,labels)

    self.log('val_loss', res.loss, prog_bar=True,batch_size=self.hparameters['batch_size'])
    return res.loss

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
    'max_epochs': 3,
    'batch_size': 4,
    'learning_rate': 2e-5,
    'train_dir': "./dataset/Wiki10-31K/train_finetune.json",
    'val_dir': "./dataset/Wiki10-31K/test_finetune.json",
    'data_dir': "./dataset/Wiki10-31K/",
    'model': 'BART'
}

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
checkpoint_callback = ModelCheckpoint(
        dirpath='./log/t2t_check',
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
    )
lr_callback = LearningRateMonitor(logging_interval="step")
model = GenerationModel(hparams)
logger = TensorBoardLogger(save_dir=os.path.join(hparams['data_dir'],'t2t'),name=hparams['model']+'_log')
if not os.path.exists(os.path.join(hparams['data_dir'],hparams['model']+'_save')):
  os.mkdir(os.path.join(hparams['data_dir'],hparams['model']+'_save'))
trainer = pl.Trainer(max_epochs=3, callbacks=[early_stopping,checkpoint_callback ,lr_callback], logger=logger,
                     #auto_lr_find=True,
                     default_root_dir=os.path.join(hparams['data_dir'],hparams['model']+'_save'),
                     accelerator="gpu", devices=1)

trainer.fit(model,train_dataloaders=model.train_dataloader(),
                val_dataloaders=model.val_dataloader())