import string 
import json
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize  
nltk.download('stopwords')

stemmer = SnowballStemmer("english")
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
p_max = 0
p_at_1_count = 0
p_at_3_count = 0
p_at_5_count = 0
#test_pegasus.same
with open("test_pegasus_combine_stem_b.txt", "r+") as pred_txt:
    pred_list = []
    #test.tgt
    with open("test_labels.txt", "r+") as tgt_txt:
        tgt_list = []
        for line in pred_txt:
            curr_pred = line.strip().split(" ")
            pred_list.append(curr_pred)
            
        for line in tgt_txt:
            curr_tgt_ori = line.split(" ")
            curr_tgt = []
            for per_label in curr_tgt_ori:
                word_list = []
                for per_word in per_label.split('_'):
                    word_list.append(stemmer2.stem(per_word))
                curr_tgt.append("_".join(word_list))
            tgt_list.append(curr_tgt)

        for i in range(len(pred_list)):
            max_flag = 0
            p_3 = 0
            for j in range(3):
                #需要修改
                if  (len(pred_list[i])> j) and (pred_list[i][j] in tgt_list[i]):
                    p_3 += 1
            p_at_3_count += p_3
            p_5 = 0
            if len(pred_list[i])<5:
                for j in range(len(pred_list[i])):
                    if pred_list[i][j] in tgt_list[i]:
                        p_5 += 1
            else:
                for j in range(5):
                    if pred_list[i][j] in tgt_list[i]:
                        p_5 += 1
            p_at_5_count += p_5
            for k in pred_list[i]:
                if k in tgt_list[i]:
                    max_flag  = 1
            p_max += max_flag
            if pred_list[i][0] in tgt_list[i]:
                # print(pred_list[i][0])
                # print('12324356')
                p_at_1_count += 1
                
            else:
                print(str(i) + " " + pred_list[i][0])
                
        print(p_at_1_count / len(pred_list))
        print(p_at_3_count / (3*len(pred_list)))
        print(p_at_5_count / (5*len(pred_list)))
        print(p_max / len(pred_list))

