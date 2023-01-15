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

datadir = "./dataset/Wiki10-31K/"
model = GenerationModel.load_from_checkpoint("./log/t2t_check/epoch=3-val_loss=0.46-other_metric=0.00.ckpt").to(device)
#model.load_from_checkpoint("./log/prefix_check/"+"epoch=2-val_loss=0.50-other_metric=0.00.ckpt").to(device)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
predict_prefix(model=model,tokenizer=tokenizer,datadir=datadir,src_document="test_texts.txt",output_dir="test_pred_t2t.txt",data_size=8,model_cate='t2t')