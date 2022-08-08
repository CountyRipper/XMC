from typing import List
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/stsb-roberta-base')
'''
这个文件的意义：
same-预测出来的词在总标签域中全词匹配到
similar-在全标签域中无法匹配到，找出最相似的一组标签(这个标签数量=预测出来的标签数量-全词匹配的数量)
combine=same+similar
preformat = 原生预测标签结果的词干化，上述的前三个也是词干化去重之后的结果
需要跑两次得到test和train的两组数据，train的用于ranking训练，test的用于结果测试
'''
scores = model.predict([('Sent A1', 'Sent B1'), ('Sent A2', 'Sent B2')])
print(type(scores))
print(scores)
# stem
from rouge import Rouge
from tqdm import tqdm
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

rouge = Rouge()
#stem化所有label
def write_list_into_txt(list_name, txt_name):
    with open(txt_name, "w+") as w:
        for each in list_name:
            curr_line = [x.replace(" ", "_") for x in each]
            w.write(" ".join(curr_line))
            w.write("\n")
#replace sapce of each labels  to _
#inpput template: franc, beef, intervent stock, aid to disadvantag group, communiti aid,
#output template: franc, beef, intervent_stock, aid_to_disadvantag_group,  
def merge_each_label(data_name,outputname=None)-> List[str]:
    result = [] #save the merged label result
    with open(data_name+".txt","r+") as f:
        for row in f:
            res_row = "" # cur raw result save
            single_label_set = row.split(", ") 
            for i in range(len(single_label_set)):
                single_label_set[i] = single_label_set[i].replace(" ","_") #replace sapce to _
                res_row+=single_label_set[i]+" " 
            result.append(res_row) #get the result set
        if outputname:
            with open(outputname+".txt","a+") as s:
                for i in result:
                    s.write(i+"\n")
        return result
                      
    """
    template: beef market_support france award_of_contract aid_to_disadvantaged_groups
    """
def stem_labels(data_name,outputname=None)->List[str]:
    # pay more attention, data_name should be merged
    with open(data_name+".txt","r+") as label_set:
        stemed_result_list=[] # to save stemed word set
        for line in label_set:
            cur_label_set = line.split(" ")# pay attention to format
            stem_result=[]
            for per_label in cur_label_set:
                word_list=[] # to save stemed words
                for per_word in per_label.split("_"):
                    word_list.append(stemmer2.stem(per_word))
                stem_result.append(" ".join(word_list))
            stemed_result_list.append(stem_result)
        if outputname:
            with open(outputname+".txt","w+") as f:
                for i in stemed_result_list:
                    f.write(i+"\n")
        return stemed_result_list
    
'''
    read all_label_domain, test_label(stemed), predict_labels(stemed), 
    compare test_label and predict_labels
'''
def getmy_similar_txt(dataset_name,reference_labe_name):
    label_sum_domian_list = [] # inclding all standard label list, however
                         

def get_similar_txt(dataset_name,reference_labe_name):
    # 录入stem过的labels
    label_stem_list = []
    same_label_list = []
    not_same_label_list = []
    similar_label_list = []
    combine_label_list = []
    pred_format_list = []
    # 所有出现过的label 包含train和test  from test_labels.txt and train_labels.txt
    '''
        all_labels_stem.txt = reference_labe_name
    '''
    with open(reference_labe_name+'.txt', "r+") as label_f:
        for line in label_f:
            label_stem_list.append(line.strip())   

    with open(dataset_name + "_pred.txt", "r+") as pred_txt:
        pred_list = []
        #test_pegasus.tgt  是test_labels.txt
        with open("test_labels.txt", "r+") as tgt_txt:
            tgt_list = []
            #数据预处理部分，如果之前已经处理则不需要，原则上实现的效果应该是切分到每个预测的词
            for line in pred_txt:
                curr_pred_ori = json.loads(line)['pred'].strip("[]").strip('\"').strip("'").strip("[]").split(",")
                curr_pred = [each.strip().strip("''") for each in curr_pred_ori]
                pred_list.append(curr_pred)
                pred_format = sorted(set(curr_pred), key=curr_pred.index)
                pred_format_list.append(pred_format)  
            #对于原生label文件的词干化？
            for line in tgt_txt:
                curr_tgt_ori = line.split(" ")
                curr_tgt = []
                for per_label in curr_tgt_ori:
                    word_list = []
                    for per_word in per_label.split('_'):
                        word_list.append(stemmer2.stem(per_word))
                    curr_tgt.append(" ".join(word_list))
                tgt_list.append(curr_tgt)

            for i in tqdm(range(len(pred_list))):
                same_list = []
                not_same_list = []
                similar_list = []
                combine_list = []
                # 处理
                # 判断是否生成了现有的label
                #如果预测生成的标签能在总标签库中找到就直接放入，没有找到的话就从标签库里面找相似的？？？
                for each_pred in pred_list[i]:
                    if each_pred in label_stem_list:
                        curr_p = each_pred.replace(" ", "_")
                        if curr_p not in same_list:
                            same_list.append(curr_p)
                    else:
                        not_same_list.append(each_pred.replace(" ", "_"))                    
                same_label_list.append(same_list)
                not_same_label_list.append(not_same_list)
                # filter(None, not_same_list)
                if len(not_same_list) == 0:
                    combine_list.extend(same_list)
                    similar_label_list.append(similar_list)   
                else:
                    for each_not_same in not_same_list:
                        # 计算rouge1+rouge2的值，排序
                        if len(each_not_same) <= 0:
                            continue
                        cal_score_list = []
                        for each_exsiting_label in label_stem_list:
                            cal_score_list.append((each_not_same, each_exsiting_label))
                        scores = model.predict(cal_score_list)
                        scores = scores.tolist()
                        most_similar_one = label_stem_list[scores.index(max(scores))] 
                        similar_list.append(most_similar_one)
                    #combine = 相等+相似
                    combine_list.extend(same_list)
                    combine_list.extend(similar_list)  
                    similar_label_list.append(similar_list)   
                combine_list_redundant = list(set(combine_list))
                combine_label_list.append(combine_list_redundant)     

            write_list_into_txt(same_label_list, dataset_name +'_samec'+'.txt') # .samec existing label
            #write_list_into_txt(not_same_label_list, dataset_name +'.notsame') 
            #write_list_into_txt(similar_label_list, dataset_name +'.similar') 
            write_list_into_txt(combine_label_list, dataset_name +'_combine'+'.txt') #.combine exisitng +similar labels
            write_list_into_txt(pred_format_list, dataset_name +'_predformat'+'.txt') #  每一个row内部去重+词干化


get_similar_txt('test_pegasus')#
#get_similar_txt('train_pegasus')
#添加一个 ‘train_pegasus’
