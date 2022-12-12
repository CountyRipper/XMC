from argparse import ArgumentParser
import os
import yaml
from model.rank_model import Rank_model,rankdata
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils.premethod import read_labels
from pytorch_lightning.callbacks import EarlyStopping
hparams = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "epochs": 10,
    "early_stop_epochs": 3,
}

def rank_train(args):
    yamlPath = os.path.join("./src/conf","EUR-Lex"+".yaml")
    paras=[]
    with open(yamlPath,'r+') as f:
        paras = yaml.load(f.read(),Loader=yaml.SafeLoader)
    print(paras["text2text"]['model_name'])
    train_dir = os.path.join(paras['datadir'],'train_finetune.json')
    valid_dir = os.path.join(paras['datadir'],'test_finetune.json')
    dataset = load_dataset('json',data_files={'train': train_dir, 'valid': valid_dir})
    train_texts, train_labels = [each for each in dataset['train']['document']], dataset['train']['summary']
    valid_texts, valid_labels = [each for each in dataset['valid']['document']], dataset['valid']['summary']
    tmp_labels =[]
    for i in train_labels:
        tmp_labels.append([j.strip("'") for j in i.strip("[]").split(", ")])
    train_labels = tmp_labels
    tmp_labels = []
    for i in valid_labels:
        tmp_labels.append([j.strip("'") for j in i.strip("[]").split(", ")])
    valid_labels = tmp_labels
    train_combine_labels = read_labels("./dataset/EUR-Lex/res/train_combine_labels_t5lbi.txt")
    valid_combine_labels = read_labels("./dataset/EUR-Lex/res/test_combine_labels_t5lbi.txt")
    
    rd_train = rankdata(train_texts,train_labels,train_combine_labels)
    rd_valid = rankdata(valid_texts,valid_labels,valid_combine_labels)
    train_dataloader = DataLoader(rd_train,batch_size=4,collate_fn=lambda x: x,shuffle=True)
    valid_dataloader = DataLoader(rd_valid,batch_size=4,collate_fn=lambda x: x,shuffle=True)
    args.num_training_samples = len(train_dataloader)
    model = Rank_model(args).to('cuda')
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams["early_stop_epochs"],
        verbose=True,
        mode="min",
        )
    trainer = pl.Trainer(
        max_epochs=hparams["epochs"],
        callbacks=[early_stop_callback],
        gpus=1,
        default_root_dir=os.path.join(paras['datadir'],'rank_c')
        )
    trainer.fit(model,train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument('--datadir', type=str, default='./dataset/EUR-Lex/',
    #                    help='dataset_dir')
    parser.add_argument('--warmup_epochs',type=int,default=2)
    parser.add_argument('--update_step',type=int,default=200)
    parser.add_argument('--bert_name',type=str,default='bert-uncased-base')
    parser.add_argument('--feature_layers',type=int,default=5)
    parser.add_argument('--dropout',type=float,default=0.2)
    parser.add_argument('--hidden_dim',type=int,default=300)
    parser.add_argument('--labels_num',type=int,default=30522)
    parser.add_argument('--batch_size',type=int,default=6)
    parser.add_argument('--max_epochs',type=int,default=6)
    parser.add_argument('--num_training_samples',type=int,default=0)
    parser.add_argument('--learning_rate',type=float,default=1e-2)
    
    args = parser.parse_args()
    rank_train(args)
    
    