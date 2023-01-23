from datasets import load_dataset
import pytorch_lightning as pl
import torch
import os
from tqdm import tqdm
from transformers import BartTokenizerFast
from prefix_model import Bart_Prefix_Model
from t2t_model import GenerationModel
from torch.utils.data import DataLoader
import wandb
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.init(project="t2t_model")
hparams_prefix_train = {
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
class prefix_args(object):
    def __init__(self) -> None:
        self.prefix_sequence_length=100
        self.prefix_mid_dim=512
        self.prefix_dropout=0.1
        self.bert_location="facebook/bart-base"
        self.model_knowledge_usage="concatenate"
        self.model_freeze_plm=0
        self.model_freeze_prefix=0
        self.model_map_description=0
        self.model_use_description=0
        self.special_tokens=None

def get_predict_prefix(model,tokenizer,documents):
    model = model
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)
    summary_ids = model.generate(inputs['input_ids'],inputs['attention_mask'], max_length = 256,top_k=10,num_beams = 5).to(device)
    pre_result=tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2)
    return pre_result
def get_predict_t2t(model,tokenizer,documents):
    model = model
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)
    summary_ids = model.generate(inputs['input_ids'],inputs['attention_mask'], max_length = 256,top_k=10,num_beams = 5).to(device)
    pre_result=tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2)
    return pre_result

def predict_prefix(model,tokenizer,datadir,src_document,output_dir,data_size,model_cate):
    src_document = os.path.join(datadir,src_document)
    output_dir = os.path.join(datadir,'res',output_dir)
    res = []
    data = []
    with open(src_document,'r') as r:
        for i in r:
            data.append(i)
    dataloader = DataLoader(data,batch_size= data_size)
    with open(output_dir,'a+') as t:
        for i in tqdm(dataloader): #range(len(data))
            if model_cate=='t2t':
                tmp_result = get_predict_t2t(model,tokenizer,i)
            else:
                tmp_result = get_predict_prefix(model,tokenizer,i)
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
if __name__ =='__main__':
    datadir = "./dataset/Wiki10-31K/"
    model = Bart_Prefix_Model(hparams=hparams_prefix_train,prefix_hparameters=prefix_args())
    model = Bart_Prefix_Model.load_from_checkpoint("log/pre_check/epoch=4-val_loss=0.47-other_metric=0.00.ckpt").to(device)
    #model = model.load_from_checkpoint("log/pre_check/epoch=4-val_loss=0.47-other_metric=0.00.ckpt").to(device)
    #model.load_from_checkpoint("./log/prefix_check/"+"epoch=2-val_loss=0.50-other_metric=0.00.ckpt").to(device)
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    predict_prefix(model=model,tokenizer=tokenizer,datadir=datadir,src_document="train_texts.txt",output_dir="train_pred_pre.txt",data_size=8,model_cate='pre')