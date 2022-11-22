import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from transformers import (AutoTokenizer,BartTokenizerFast, BartForConditionalGeneration,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,PegasusForConditionalGeneration, PegasusTokenizerFast,
                          T5ForConditionalGeneration,T5TokenizerFast,AutoModelForSeq2SeqLM)
import pytorch_lightning as pl

class KGXMCmodel(pl.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        if self.modelname == 'bart-large' or self.modelname == 'BART-large' or self.modelname == 'Bart-large':
            self.KG_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", cache_dir='./models').to(
                self.device)
            self.tokenizer = BartTokenizerFast.from_pretrained(pretrained_model_name_or_path="facebook/bart-large",
                                                           cache_dir='./models')
        elif self.modelname == 'bart' or self.modelname == 'BART' or self.modelname == 'Bart':
            self.KG_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", cache_dir='./models').to(
                self.device)
            self.tokenizer = BartTokenizerFast.from_pretrained(pretrained_model_name_or_path="facebook/bart-base",
                                                           cache_dir='./models')
        elif self.modelname == 'pegasus' or self.modelname == 'Pegasus':
            self.KG_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large',
                                                                         cache_dir='./models').to(self.device)
            self.tokenizer = PegasusTokenizerFast.from_pretrained(pretrained_model_name_or_path="facebook/bart-large",
                                                              cache_dir='./models')
        elif self.modelname == 'pegasus-xsum' or self.modelname == 'Pegasus-xsum':
            self.KG_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum',
                                                                         cache_dir='./models').to(self.device)
            self.tokenizer = PegasusTokenizerFast.from_pretrained(pretrained_model_name_or_path="facebook/bart-large",
                                                              cache_dir='./models')
        elif self.modelname == 't5' or self.modelname == 'T5':
            self.KG_model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base", cache_dir='./models').to(
                self.device)
            self.tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-base", cache_dir='./models')
        elif self.modelname == 't5-large' or self.modelname == 'T5-large':
            self.KG_model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large", cache_dir='./models').to(
                self.device)
            self.tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-large", cache_dir='./models')
        elif self.modelname == 'keybart' or self.modelname == 'KeyBART':
            self.tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART", cache_dir='./models')
            self.KG_model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART", cache_dir='./models').to(
                self.device)
        
            
        
