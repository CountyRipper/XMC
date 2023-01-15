import os
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from prefix_model import Bart_Prefix_Model
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer,BarthezTokenizerFast
from prompt.modeling_auto import AutoModelForSeq2SeqLM
from t2t_model import GenerationModel
from unified.prefixtuning import PrefixModel
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
hparams_prefix = {
    'max_epochs': 5,
    'batch_size': 4,
    'learning_rate': 5e-5,
    'warmup': 100,
    'max_iters': 3000,
    'train_dir': "./dataset/Wiki10-31K/train_finetune.json",
    'val_dir': "./dataset/Wiki10-31K/test_finetune.json",
    'data_dir': "./dataset/Wiki10-31K/",
    'model': 'BART'
}
hparams_t2t = {
    'max_epochs': 4,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'train_dir': "./dataset/Wiki10-31K/train_finetune.json",
    'val_dir': "./dataset/Wiki10-31K/test_finetune.json",
    'data_dir': "./dataset/Wiki10-31K/",
    'model': 'BART'
}
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


def token_data(texts,labels,tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = MyData(encodings, decodings)
    return dataset_tokenized
def train_pre(hparams):
    model = Bart_Prefix_Model(hparams)
    logger = TensorBoardLogger(save_dir=os.path.join(hparams['data_dir'],'prefix'),name=hparams['model']+'_log')

    trainer = pl.Trainer(max_epochs=3, callbacks=[early_stopping,checkpoint_callback,lr_callback], logger=logger,
                     default_root_dir=os.path.join(hparams['data_dir'],hparams['model']+'_save'),
                     enable_checkpointing=True,
                     
                     #auto_lr_find=True,
                     accelerator="gpu", devices=1)
    trainer.fit(model,train_dataloaders=model.train_dataloader(),
                val_dataloaders=model.val_dataloader())
def train_pre_hf(datadir,model,checkdir,output,batch_size,epoch):
    train_dir= datadir+"train_finetune.json"
    valid_dir= datadir+"test_finetune.json"
    output = os.path.join(datadir,'prefix_save')
    checkdir = os.path.join(datadir,'prefix_check')
    print('checkdir:'+checkdir)
    print('save_dir:'+output)
    print('batch_size:',batch_size)
    print('epoch:',epoch)       
    from datasets import load_dataset
    #prefix = "summarize: "
    dataset = load_dataset('json',data_files={'train': train_dir, 'valid': valid_dir}).shuffle(seed=42)
    train_texts, train_labels = [each for each in dataset['train']['document']], dataset['train']['summary']
    valid_texts, valid_labels = [each for each in dataset['valid']['document']], dataset['valid']['summary']
    tokenizer = BarthezTokenizerFast.from_pretrained(model.prefix_hparameters.bert_location)
    train_dataset = token_data(train_texts,train_labels,tokenizer)
    valid_dataset = token_data(valid_texts,valid_labels,tokenizer)
    train_args = Seq2SeqTrainingArguments(
        output_dir=checkdir,
        num_train_epochs=epoch,           # total number of training epochs
        per_device_train_batch_size=batch_size,   # batch size per device during training, can increase if memory allows
        per_device_eval_batch_size=batch_size,    # batch size for evaluation, can increase if memory allows
        save_steps=30000,                  # number of updates steps before checkpoint saves
        save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
        evaluation_strategy = "epoch",     # evaluation strategy to adopt during training                 # number of update steps before evaluation
        learning_rate= 5e-5,  # learning rate
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        predict_with_generate=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=train_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,            # evaluation dataset
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(output)

def train_t2t(hparams):
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
    trainer = pl.Trainer(max_epochs=4, callbacks=[early_stopping,checkpoint_callback ,lr_callback], logger=logger,
                         #auto_lr_find=True,
                         default_root_dir=os.path.join(hparams['data_dir'],hparams['model']+'_save'),
                         accelerator="gpu", devices=1)

    trainer.fit(model,train_dataloaders=model.train_dataloader(),
                    val_dataloaders=model.val_dataloader())
if __name__ =="__main__":
    #prefix_hparameters = prefix_args()
    #model = PrefixModel(prefix_hparameters).to(device)
    #model.load_from_checkpoint("./log/prefix_check/"+"epoch=2-val_loss=0.50-other_metric=0.00.ckpt")
    #model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    #training(datadir="./dataset/Wiki10-31K/",model=model,checkdir='prefix_check',output='prefix_save',
             #batch_size=4,epoch=4)
    train_t2t(hparams_t2t)