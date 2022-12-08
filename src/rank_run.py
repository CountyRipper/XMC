import os
import yaml
from rank_model import Rank_model,rankdata
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
yamlPath = os.path.join("./src/conf","EURLex-4K"+".yaml")
paras=[]
with open(yamlPath,'r+') as f:
    paras = yaml.load(f.read(),Loader=yaml.SafeLoader)
print(paras["text2text"]['model_name'])
train_dir = os.path.join(paras['datadir'],'train_finetune.json')
valid_dir = os.path.join(paras['datadir'],'test_finetune.json')
dataset = load_dataset('json',data_files={'train': train_dir, 'valid': valid_dir}).shuffle(seed=42)
train_texts, train_labels = [each for each in dataset['train']['document']], dataset['train']['summary']
valid_texts, valid_labels = [each for each in dataset['valid']['document']], dataset['valid']['summary']
model = Rank_model()
rd_train = rankdata(train_texts,train_labels)
rd_valid = rankdata(valid_texts,valid_labels)
train_dataloader = DataLoader(rd_train,batch_size=16)
valid_dataloader = DataLoader(rd_valid,batch_size=16)
trainer = pl.Trainer()
trainer.fit(model,train_dataloaders=train_dataloader,
            valid_dataloaders=valid_dataloader,)
