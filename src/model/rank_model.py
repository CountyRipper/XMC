from pickletools import optimize
import numpy as np
import torch
from torch import nn

from transformers import BertTokenizerFast, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
import pytorch_lightning as pl
from tokenizers import BertWordPieceTokenizer


def get_bert(bert_name):
    if 'roberta' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained('roberta-base', config=model_config)
    elif 'xlnet' in bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained('bert-base-uncased')
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
    return bert
class CosineWarmupScheduler(optimize.lr_scheduler._LRScheduler):
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
参数：bert名字， labels_num：是all_labels数量
labels_num,bert='bert-base-uncased',feature_layers=5,
                 dropout =0.5, update_count=1,candidates_topk=10,
                 use_swa=True,warmup_epoch=10,update_step=200,hidden_dim=300
'''
class rank_model(pl.LightningModule):
    def __init__(self, args) -> None:
        
        super().__init__()
        self.bert  = get_bert()
        self.warmup_epoch = args.warmup_epoch
        self.update_step = args.update_step
        
        self.bert_name, self.bert = args.bert_name, get_bert(args.bert_name)
        self.feature_layers, self.drop_out = args.feature_layers, nn.Dropout(args.dropout)
        #hiden bottleneck layer
        self.liner0 = nn.Linear(self.feature_layers*self.bert.config.hidden_size,args.hidden_dim)
        self.embed = nn.Embedding(args.labels_num, args.hidden_dim)
        nn.init.xavier_uniform_(self.embed.weight)#初始化权重均匀分布
    '''
    对于模型，需要的数据有，text，cmobined_labels作为输入, truth_labels提供最优顺序
    首先处理数据通过combined_labels的输入，每一个label对照truth_labels的顺序排序：先抽取truth包含的label然后把这些全部排到前面
    两种方式（测试）：1.不改变combined_labels内部排序，仅将truth中有的全部移动到前面
    2.完全按照truth中的顺序
    candidates 可以是上一步
    '''
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,
                candidates=None):
        # candidates是待排序的labels集合，直接由上一步地combined给出
        input_ids
        loss_fn = torch.nn.BCEWithLogitsLoss()#定义loss函数
        #获取最后一层输出
        bert_outs = self.bert(input_ids,attention_mask = attention_mask,
                              token_type_ids=token_type_ids)[-1]
        #按次获取倒数第一到第五层的[cls],cls是用0取出来的
        out = torch.cat([bert_outs[-i][:,0] for i in range(1,self.feature_layers+1)],dim=-1)
        out = self.drop_out(out)
        #根据candidates来获取得分矩阵
        
        
        #线性层获取文本表示 text representation
        emb = self.liner0(out) 
        embed_weights = self.embed(candidates) # N, sampled_size, H
        emb = emb.unsqueeze(-1)
        logits = torch.bmm(embed_weights,emb).squeeze(-1)
        return logits, candidates, candidates_score
    def configure_optimizers(self):
        base_opt = torch.optim.Adam(self.parameters(),lr=1e-5)
        return base_opt
    #batch 包含labels和texts
    def training_step(self, batch, batch_idx):
        text, labels = batch
        #self应该输入 enconding之后的
        ranked_labels = self(text,labels)
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def test_step(self, batch, batch_idx):
        pass
    