import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch
from transformers import (AutoTokenizer, BartTokenizerFast, BartForConditionalGeneration, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, PegasusForConditionalGeneration, PegasusTokenizerFast,
                          T5ForConditionalGeneration, T5TokenizerFast, AutoModelForSeq2SeqLM)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
wandb.init(project="t2t_model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyData(torch.utils.data.Dataset):
    def __init__(self, encoding, labels):
        self.ids = encoding['input_ids']
        self.mask = encoding['attention_mask']
        self.labels = labels['input_ids']

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.tensor(self.ids[idx]).to(device)
        item['attention_mask'] = torch.tensor(self.mask[idx]).to(device)
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        #item={'input_ids': torch.tensor(val[idx]).to(device) for key, val in self.encoding.items()}
        #item['labels'] = torch.tensor(self.labels['input_ids'][idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)  # len(self.labels)


class GenerationModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.hparameters = hparams
        self.curr_avg_loss = 0.0
        if hparams == None:
            self.tokenizer = BartTokenizerFast.from_pretrained(
                "facebook/bart-base")
            self.model = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base").to(device)
        else:
            if ('bart' in hparams['model']) or ('Bart' in hparams['model']) or ('BART' in hparams['model']):
                if ('bart-large' in hparams['model']) or ('Bart-large' in hparams['model']) or ('BART-large' in hparams['model']):
                    self.tokenizer = BartTokenizerFast.from_pretrained(
                        "facebook/bart-large")
                    self.model = BartForConditionalGeneration.from_pretrained(
                        "facebook/bart-large").to(device)
                else:
                    self.tokenizer = BartTokenizerFast.from_pretrained(
                        "facebook/bart-base")
                    self.model = BartForConditionalGeneration.from_pretrained(
                        "facebook/bart-base").to(device)
            elif ('pegasus' in hparams['model']) or ('Pegasus' in hparams['model']):
                self.tokenizer = PegasusTokenizerFast.from_pretrained(
                    'google/pegasus-large')
                self.model = PegasusForConditionalGeneration.from_pretrained(
                    'google/pegasus-large').to(device)
            elif ('t5' in hparams['model']) or ('T5' in hparams['model']):
                if ('t5-large' in hparams['model']) or ('T5-large' in hparams['model']):
                    self.tokenizer = T5TokenizerFast.from_pretrained(
                        'google/t5-v1_1-large')
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        'google/t5-v1_1-large').to(device)
                else:
                    self.tokenizer = T5TokenizerFast.from_pretrained(
                        'google/t5-v1_1-base')
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        'google/t5-v1_1-base').to(device)
        self.save_hyperparameters()
    def forward(self, encoder_input_ids, decoder_input_ids):
        return self.model(input_ids=encoder_input_ids, labels=decoder_input_ids)

    def training_step(self, batch, batch_idx):
        encoder_input_ids, encoder_attention_mask, labels = torch.stack([i['input_ids'] for i in batch]), torch.stack(
            [i['attention_mask'] for i in batch]), torch.stack([i['labels'] for i in batch])
        res = self(encoder_input_ids, labels)
        # loss = self.custom_loss(logits, labels) custom loss
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        global_step = self.trainer.global_step
        # #手动优化scheduler
        sch = self.lr_schedulers()
        loss = res.loss
        self.curr_avg_loss+=loss
        if (global_step+1)%50 == 0:
            wandb.log({"loss": self.curr_avg_loss/50,"global_step":global_step})
            wandb.log({"learning_rate":cur_lr,"global_step":global_step})
            wandb.log({"train_epoch":self.trainer.current_epoch,"global_step":global_step})
            self.curr_avg_loss= 0.0
        if (batch_idx + 1) % 5 == 0:
            sch.step()
            
        self.log('lr',cur_lr, prog_bar=True, on_step=True)
        #self.log('train_loss', loss, prog_bar=True,batch_size=self.hparameters['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        encoder_input_ids, encoder_attention_mask, labels = torch.stack([i['input_ids'] for i in batch]), torch.stack(
            [i['attention_mask'] for i in batch]), torch.stack([i['labels'] for i in batch])
        #encoder_input_ids, encoder_attention_mask,labels = batch['input_ids'],batch['attention_mask'],batch['labels']
        res = self(encoder_input_ids, labels)
        #loss, logits = self(encoder_input_ids,labels)

        self.log('val_loss', res.loss, prog_bar=True,batch_size=self.hparameters['batch_size'])
        return res.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparameters['learning_rate'])
        # self.lr_scheduler = CosineWarmupScheduler(
        #     optimizer, warmup=self.hparameters['warmup'], max_iters=self.hparameters['max_iters']
        # )
        # return optimizer
        return{
            "optimizer": optimizer,
            #"lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100),
            "lr_scheduler":torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.999),
            "interval": "step",
            "frequency": 1,
        }
    def generate(self, input_ids, attention_mask, max_length, top_k, num_beams):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   max_length=max_length, top_k=top_k, num_beams=num_beams)

    def train_dataloader(self):
        datadir = self.hparameters['train_dir']
        prefix = "summarize: "
        dataset = load_dataset('json', data_files=datadir)
        train_texts, train_labels = [
            prefix + each for each in dataset['train']['document']], dataset['train']['summary']

        encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        decodings = self.tokenizer(train_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        train_data = DataLoader(
            dataset_tokenized, batch_size=self.hparameters['batch_size'], collate_fn=lambda x: x, shuffle=True)
        # create a dataloader for your training data here
        return train_data

    def val_dataloader(self):
        datadir = self.hparameters['val_dir']
        prefix = "summarize: "
        dataset = load_dataset('json', data_files=datadir)
        val_texts, val_labels = [
            prefix + each for each in dataset['train']['document']], dataset['train']['summary']

        encodings = self.tokenizer(val_texts, truncation=True, padding=True)
        decodings = self.tokenizer(val_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        print(len(dataset_tokenized))
        val_data = DataLoader(
            dataset_tokenized, batch_size=self.hparameters['batch_size'], collate_fn=lambda x: x, shuffle=True)
        # create a dataloader for your training data here
        return val_data
