# import string 
# import json
# import nltk
# from nltk.stem import *
# from nltk.stem.porter import *
# from nltk.stem.snowball import SnowballStemmer
# from nltk.tokenize import word_tokenize  
from sentence_transformers.cross_encoder import CrossEncoder
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

# nltk.download('stopwords')

# stemmer = SnowballStemmer("english")
# stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
def rank_train(dir,text_data,train_pred_data,train_label_data,model_save_dir):
    
    fine_tune_list = []
    raw_text_list = []
    label_list=[]
    pred_label_list=[]
    label_score_list=[]  
    text_src=dir+text_data
    train_pred_src=dir+train_pred_data
    train_label_src= dir+train_label_data
    model_save_dir = dir+model_save_dir
    print("text_data: "+text_src)
    print("train_pred_data"+train_pred_src)
    print("train_label_src:"+train_label_src)
    print("model_save_dir: "+model_save_dir)
    with open(text_src, "r+") as raw_text: ## train_text
        for line in raw_text:
            raw_text_list.append(line.strip()) # strip \n
    # train_text 生成出来的-pred预测（没有找mactch，没有combine）stem化 
    with open(train_pred_src, "r+") as pred_txt: #train_pred.txt
        for row in pred_txt:
            pred_label_list.append(row.strip().split(", "))
    with open(train_label_src, "r+") as label_txt: #train_labels_stem.txt
        for row in label_txt:
            label_list.append(row.strip().split(", "))
    print(str(raw_text_list[0])+'\n'+str(pred_label_list[0])+'\n'+str(label_list[0]))
    # with open(train_pred_src, "r+") as pred_txt:
    #     build_label_list = []
    #     pred_list = []
    #     #train_labels.txt
    #     with open(train_label_src, "r+") as tgt_txt:
    #         tgt_list = []
    #         for line in pred_txt:
    #             curr_pred_ori = line.split(" ")
    #             curr_pred = [each.strip().replace("_", " ") for each in curr_pred_ori]
    #             pred_list.append(curr_pred)
    #         # train_labels_stem.txt
    #         for line in tgt_txt:
    #             curr_tgt_ori = line.strip().split(" ")
    #             curr_tgt = []
    #             for per_label in curr_tgt_ori:
    #                 word_list = []
    #                 for per_word in per_label.split('_'):
    #                     word_list.append(stemmer2.stem(per_word))
    #                 curr_tgt.append(" ".join(word_list))
    #             tgt_list.append(curr_tgt)
            #核心部分：预测出来的标签跟原来的标签作对比，计算位置顺序  
    for i in range(len(pred_label_list)):
        for each in pred_label_list[i]:
            label_len_list = len(label_list[i])
            if each in label_list[i]:
                label_score = 0.5+0.5 *(label_len_list - label_list[i].index(each))/label_len_list
                #label_score = 1
                fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each], label=label_score))
                label_score_list.append(str(i) + ' ' + each+ ' ' +str(label_score))
            else:
                fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each], label=0))
                label_score_list.append(str(i) + ' ' + each+ ' 0')
            '''
            for i in range(len(pred_list)):
                for each_pred in pred_list[i]:
                    if each_pred in tgt_list[i]:
                        fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each_pred], label=1))
                    else:
                        fine_tune_list.append(InputExample(texts=[raw_text_list[i].rstrip(), each_pred], label=0))
            '''
    with open (dir+"build_labels.txt", "w+") as fb:
        for each in label_score_list:
            fb.write(each)
            fb.write("\n")    
    #print('file complete')

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
              output_path=model_save_dir)
    model.save(model_save_dir)

#rank_train('./dataset/EUR-Lex/',"train_texts.txt","generate_result/train_pred.txt","train_labels.txt","cr_en")

