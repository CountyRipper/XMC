import string 
import json
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize  
from sentence_transformers.cross_encoder import CrossEncoder
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

nltk.download('stopwords')

stemmer = SnowballStemmer("english")
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)

p_at_1_count = 0

fine_tune_list = []
dev_samples = []
raw_text_list = []
test_list = []
prefix = ''
# train_texts.txt 
with open("train_texts.txt", "r+") as raw_text:
    for line in raw_text:
        raw_text_list.append(line.strip())
# train_text 生成出来的-pred预测（没有找mactch，没有combine）stem化 去重
with open("train_pegasus_predformat.txt", "r+") as pred_txt:
    build_label_list = []
    pred_list = []
    #train_labels.txt
    with open("train_labels.txt", "r+") as tgt_txt:
        tgt_list = []
        for line in pred_txt:
            curr_pred_ori = line.split(" ")
            curr_pred = [each.strip().replace("_", " ") for each in curr_pred_ori]
            pred_list.append(curr_pred)
            
        for line in tgt_txt:
            curr_tgt_ori = line.strip().split(" ")
            curr_tgt = []
            for per_label in curr_tgt_ori:
                word_list = []
                for per_word in per_label.split('_'):
                    word_list.append(stemmer2.stem(per_word))
                curr_tgt.append(" ".join(word_list))
            tgt_list.append(curr_tgt)
            
        for i in range(len(pred_list)):
            for each in pred_list[i]:
                len_tgt_list = len(tgt_list[i])
                if each in tgt_list[i]:
                    label_score = 0.5+0.5 *(len_tgt_list - tgt_list[i].index(each))/len_tgt_list
                    #label_score = 1
                    fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each], label=label_score))
                    build_label_list.append(str(i) + ' ' + each+ ' ' +str(label_score))
                else:
                    fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each], label=0))
                    build_label_list.append(str(i) + ' ' + each+ ' 0')
        '''
        for i in range(len(pred_list)):
            for each_pred in pred_list[i]:
                if each_pred in tgt_list[i]:
                    fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each_pred], label=1))
                else:
                    fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each_pred], label=0))
        '''
     
with open ("build_labels.txt", "w+") as fb:
    for each in build_label_list:
        fb.write(each)
        fb.write("\n")    
        
print('file complete')
        
num_epoch = 5

model = CrossEncoder('cross-encoder/stsb-roberta-base', num_labels=1)

train_dataloader = DataLoader(fine_tune_list, shuffle=True, batch_size=16)
# shuffle=True



# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epoch * 0.1) #10% of train data for warm-up
#logger.info("Warmup-steps: {}".format(warmup_steps))

model.fit(train_dataloader=train_dataloader,
          epochs=num_epoch,
          warmup_steps=warmup_steps,
          #用curr
          output_path="./curr")
model.save("./curr")

