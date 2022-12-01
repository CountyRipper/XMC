import os
from typing import List
from sentence_transformers.cross_encoder import CrossEncoder
# import nltk
# from nltk.stem import *
# from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def rank(dir,text_dir,pred_combine_dir,model_dir,outputdir=None)-> List[List[str]]:
    text_dir=dir+text_dir
    pred_combine_dir=os.path.join(dir,'res',pred_combine_dir)
    model_dir=dir+model_dir
    outputdir = os.path.join(dir,'res',outputdir)
    print('rank processing:'+'\n')
    print('text_dir: '+text_dir)
    print('pred_combine_dir: '+pred_combine_dir)
    print('rank_model_dir: '+model_dir)
    print('outputdir:', outputdir)
    model = CrossEncoder(model_dir)
    pred_label_list=[]
    text_list=[]
    ranked_list=[]#保存排序好的列表
    scores_list=[]#记录分数列表的列表
    with open(pred_combine_dir, 'r+')as pre_file:
        for row in pre_file:
            pred_label_list.append(row.strip().split(", "))
    with open(text_dir,'r+') as text_file:
        for row in text_file:
            text_list.append(row.strip())
    num1 = len(pred_label_list)
    num2 = len(text_list)
    if num1!=num2:
        print('src_value error')
        return
    for i in tqdm(range(num1)):
        score_list=[]
        #ranked_list=[]
        src_text = text_list[i]
        cur_label_set = pred_label_list[i]
        pairs = []
        for each_label in cur_label_set:
            pairs.append([src_text,each_label])
            #score = model.predict([src_text,each_label])
            #score_list.append((each_label,score))
        scores = model.predict(pairs)
        scores_list = zip(scores,cur_label_set)
        
        if i%1000==0:
            print(scores_list)
        #scores_list.sort(key= lambda x:x[1],reverse=True) #按照分数排序
        scores_list = sorted(scores_list,reverse=True)
        if i%1000==0:
            print(scores_list)
        #scores_list.append(scores)
        ranked_list.append(list(map(lambda x:x[1],scores_list)))#抽取label部分
    if outputdir:
        with open(outputdir,'w+') as w1:
            for row in ranked_list:
                w1.write(", ".join(row)+'\n')
    with open(outputdir.rstrip(".txt")+"_score.txt",'w+') as w2:
        for row in scores_list:
            w2.write(str(row)+'\n')                   
    return ranked_list

def rank_bi(dir,text_dir,pred_combine_dir,model_dir,outputdir=None)-> List[List[str]]:
    text_dir=dir+text_dir
    pred_combine_dir=os.path.join(dir,'res',pred_combine_dir)
    model_dir=dir+model_dir
    outputdir = os.path.join(dir,'res',outputdir)
    print('rank processing:'+'\n')
    print('text_dir: '+text_dir)
    print('pred_combine_dir: '+pred_combine_dir)
    print('rank_model_dir: '+model_dir)
    print('output:',outputdir)
    #model_c = CrossEncoder('cross-encoder/stsb-roberta-base')
    #model_b = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer(model_dir)
    pred_label_list=[]
    text_list=[]
    ranked_list=[]#保存排序好的列表
    scores_list=[]#记录分数列表的列表
    with open(pred_combine_dir, 'r+')as pre_file:
        for row in pre_file:
            pred_label_list.append(row.strip().split(", "))
    with open(text_dir,'r+') as text_file:
        for row in text_file:
            text_list.append(row.strip())
    num1 = len(pred_label_list)
    num2 = len(text_list)
    if num1!=num2:
        print('src_value error')
        return
    for i in tqdm(range(num1)):
        score_list=[]
        #ranked_list=[]
        src_text = text_list[i]
        cur_label_set = pred_label_list[i]
        #获取文本和不同标签的embedding
        text_embedding = model.encode(src_text,convert_to_tensor=True)
        label_embedding = model.encode(cur_label_set,convert_to_tensor=True)
        cosine_scores = util.cos_sim(text_embedding, label_embedding)
        #获取之后计算得分
        #cosine_scores应该是一个数组，对应每个标签的优先级
        for ind,each_score in enumerate(cosine_scores[0].tolist()):
            score_list.append([cur_label_set[ind],each_score])
        if i%1000==0:
            print(score_list)
        score_list.sort(key= lambda x:x[1],reverse=True) #按照分数排序
        if i%1000==0:
            print(score_list)
        scores_list.append(score_list)
        ranked_list.append(list(map(lambda x:x[0],score_list)))#抽取label部分
    if outputdir:
        with open(outputdir,'w+') as w1:
            for row in ranked_list:
                w1.write(", ".join(row)+'\n')
    with open(outputdir.rstrip(".txt")+"_score.txt",'w+') as w2:
        for row in scores_list:
            w2.write(str(row)+'\n')                   
    return ranked_list
