
from functools import partial
from pickletools import optimize
import numpy as np
import torch
from torch import nn

from transformers import BertTokenizerFast, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
from transformers import XLNetTokenizerFast, XLNetModel, XLNetConfig
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from tokenizers import BertWordPieceTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)

class pred_data(torch.utils.data.Dataset):
    def __init__(self,texts,combine_labels) :
        super().__init__()
        self.texts = texts
        self.combine_labels = combine_labels
    def __getitem__(self,index):
        item={}
        item['text'] = self.texts[index]
        item['combine_labels'] = self.combine_labels[index]
        return item
    def __len__(self):
        return len(self.texts)


class rankdata(torch.utils.data.Dataset):
    def __init__(self,texts,combine_labels,gt_labels=None) -> None:
        super().__init__()
        self.texts = texts
        self.combine_labels = combine_labels
        self.gt_labels = gt_labels
    def __getitem__(self,index):
        item={}
        item['text'] = self.texts[index]
        item['combine_labels'] = self.combine_labels[index]
        item['gt_labels'] = self.gt_labels[index]
        return item
    def __len__(self):
        return len(self.texts)


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

'''
?????????bert????????? labels_num??????all_labels??????
labels_num,bert='bert-base-uncased',feature_layers=5,
                 dropout =0.5, update_count=1,candidates_topk=10,
                 use_swa=True,warmup_epoch=10,update_step=200,hidden_dim=300
'''
class Rank_model(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.update_step = args.update_step
        self.bert_name  = args.bert_name
        self.learning_rate = args.learning_rate
        self.tokenizer =self.get_tokenizer()
        self.bert = self.get_bert()
        self.save_hyperparameters()
        self.hidden_dim = args.hidden_dim
        self.warmup_epochs = args.warmup_epochs
        self.max_epochs = args.max_epochs
        self.train_iters_per_epoch = args.num_training_samples // args.batch_size
        self.feature_layers, self.drop_out = args.feature_layers, nn.Dropout(args.dropout)
        #hiden bottleneck layer
        self.liner0 = nn.Linear(self.feature_layers*self.bert.config.hidden_size,args.hidden_dim)
        self.liner1 = nn.Linear(self.bert.config.hidden_size,args.hidden_dim)
        self.loss_line = []
        self.cur_batch = []
        self.sum_loss = 0.0
        self.mean_loss= 0.0
        #self.embed = nn.Embedding(args.labels_num, args.hidden_dim)
        #nn.init.xavier_uniform_(self.embed.weight)#???????????????????????????
    def get_bert(self):
        if 'roberta' in self.bert_name:
            print('load roberta-base')
            model_config = RobertaConfig.from_pretrained('roberta-base')
            model_config.output_hidden_states = True
            bert = RobertaModel.from_pretrained('roberta-base', config=model_config).to(device)
        elif 'xlnet' in self.bert_name:
            print('load xlnet-base-cased')
            model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
            model_config.output_hidden_states = True
            bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config).to(device)
        else:
            print('load bert-base-uncased')
            model_config = BertConfig.from_pretrained('bert-base-uncased')
            model_config.output_hidden_states = True
            bert = BertModel.from_pretrained('bert-base-uncased', config=model_config).to(device)
        return bert    
    def get_tokenizer(self):
        if 'roberta' in self.bert_name:
            print('load roberta-base tokenizer')
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.bert_name:
            print('load xlnet-base-cased tokenizer')
            tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
        else:
            print('load bert-base-uncased tokenizer')
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        return tokenizer
    '''
    ????????????????????????????????????text???cmobined_labels????????????, truth_labels??????????????????
    ????????????????????????combined_labels?????????????????????label??????truth_labels???????????????????????????truth?????????label?????????????????????????????????
    ???????????????????????????1.?????????combined_labels?????????????????????truth??????????????????????????????
    2.????????????truth????????????
    candidates ??????????????????
    '''
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,
                candidates=None):
        # candidates???????????????labels??????????????????????????????combined??????
        #????????????????????????
        bert_outs = self.bert(input_ids,attention_mask = attention_mask,
                              token_type_ids=token_type_ids)[-1]
        #???????????????????????????????????????[cls],cls??????0????????????
        out = torch.cat([bert_outs[-i][:,0] for i in range(1,self.feature_layers+1)],dim=-1)
        out = self.drop_out(out)
        
        #??????candidates?????????????????????candidates_score
        labels_outs = self.bert(**candidates).pooler_output
        
        #??????????????????????????? text representation
        labels_outs = self.liner1(labels_outs)
        text_emb = self.liner0(out) #emb??????????????????????????????bottleck hidden???????????????
        labels_outs = labels_outs.unsqueeze(-1)
        #print("emb:", text_emb.size())
        #print('labels_outs:',labels_outs.size())
        #embed_weights = self.embed(candidates) # N, sampled_size, H
        text_emb = text_emb.unsqueeze(-1)
        labels_outs=labels_outs.reshape(1,len(candidates['input_ids']),self.hidden_dim)
        #print("emb:", text_emb.size())
        #print('labels_outs:',labels_outs.size())

        logits = torch.bmm(labels_outs,text_emb).squeeze(-1)
        return logits
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        return [optimizer]
        # warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         linear_warmup_decay(warmup_steps),
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        # return [optimizer], [scheduler]
        # return base_opt
    #
    def shared_step(self, batch):
        loss_s = []
        loss_fn = torch.nn.BCELoss()#??????loss??????
        for i in range(len(batch)):
            texts, gt_labels, combine_labels = batch[i]['text'],batch[i]['gt_labels'],batch[i]['combine_labels']
            #self???????????? enconding?????????
            inputs = self.tokenizer(texts,padding=True,return_tensors='pt',truncation=True).to(self.device)
            
            candidates = self.tokenizer(combine_labels, padding=True,return_tensors='pt').to(self.device)
            logits = self(inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids']
                          ,candidates)
            #print('logist:')
            #print(logits[0])
            sig = torch.nn.Sigmoid()
            pre_scores = sig(logits).reshape(len(combine_labels))
            #print(pre_scores)
            num = len(combine_labels)
            scores = torch.zeros(num).to(self.device)
            for i in range(num):
                if combine_labels[i] in gt_labels:
                    scores[i] = 1.0
            #gt_labels ???????????????
            #??????????????????combine????????????????????????gt_labels?????????????????????????????????
            loss_s.append(loss_fn(pre_scores,scores))

        loss = torch.stack(loss_s).to(self.device).mean()
        return loss
    '''
    batch ??????labels???texts. ???trian????????????batch??????combine_labels???gt_labels
    '''
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        # self.loss_line.append(loss.item())
        # self.cur_batch.append(batch_idx)
        # plt.plot(self.cur_batch,self.loss_line)
        # plt.show()
        self.sum_loss=self.mean_loss*batch_idx+loss.item()
        self.mean_loss = self.sum_loss/(batch_idx+1)
        if(batch_idx%50==0):
            with open("./log/rank.log",'a+') as f:
                s1 = 'batch_idx: '+str(batch_idx)+' loss: {:.4f}'.format(self.mean_loss)+" lr: {:.4f}".format(self.learning_rate)+'\n'
                f.write(s1)
            print('loss:%f, lr:%f'%(self.mean_loss,self.learning_rate))
        self.log("train_loss", loss, on_step=True,batch_size=len(batch),logger=True)
        return {'loss':loss,'lr':self.learning_rate}
    '''
    same as training_step + acc or loss
    '''
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=True,batch_size=len(batch),logger=True)
    '''
    batch just contains combine labels, output 
    '''
    def predict_step(self, batch, batch_idx):
        texts, combine_labels = batch[batch_idx]['text'],batch[batch_idx]['combine_labels']
        #self???????????? enconding?????????
        inputs = self.tokenizer(texts,padding=True,return_tensors='pt',truncation=True).to(self.device)
        candidates = self.tokenizer(combine_labels, padding=True,return_tensors='pt').to(self.device)
        logits = self(inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids']
                      ,candidates['input_ids'])
        pre_scores = torch.nn.Sigmoid(logits)

        return pre_scores
    
    