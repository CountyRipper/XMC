from typing import List
from sentence_transformers.cross_encoder import CrossEncoder
# import nltk
# from nltk.stem import *
# from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

def rank(dir,text_dir,pred_combine_dir,model_dir,outputdir=None)-> List[List[str]]:
    text_dir=dir+text_dir
    pred_combine_dir=dir+pred_combine_dir
    model_dir=dir+model_dir
    outputdir = dir+outputdir
    print('rank processing:'+'\n')
    print('text_dir: '+text_dir)
    print('pred_combine_dir: '+pred_combine_dir)
    print('rank_model_dir: '+model_dir)
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
        for each_label in cur_label_set:
            score = model.predict([src_text,each_label])
            score_list.append((each_label,score))
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
# nltk.download('stopwords')

# stemmer = SnowballStemmer("english")
# stemmer2 = SnowballStemmer("english", ignore_stopwords=True)

# model = CrossEncoder('./curr')

# same_list = []
# src_list = []
# # 读取same  test_pegasus.same
# with open("test_pegasus_predformat_stem_b.txt", "r+") as same_f:
#     for line in same_f:
#         same_list.append(line.strip().split(" "))

# # 建立result list存结果
# result_list = []
# scores_list = []
# # 读取src 排序 test.src 
# with open("test_texts.txt", "r+") as src_f:
#     for line in src_f:
#         src_list.append(line.strip())

# for i in tqdm(range(len(same_list))):
    
#     score_list = []
#     rank_list = []
#     res_list = []
    
#     candidates = same_list[i]
#     src_text = src_list[i]
#     for each_candidate in candidates:
#         candidate = each_candidate.replace("_", " ")
#         score = model.predict([src_text, candidate])
#         score_list.append(score)
#         rank_list.append(score)
#     rank_list.sort(reverse=True)
#     for each_res in rank_list:
#         res_list.append(candidates[score_list.index(each_res)])
#     result_list.append(res_list)
#     scores_list.append([str(x) for x in rank_list])

# with open("rank_eurlex4k.txt", "w+") as rank_f:
#     for each in result_list:
#         rank_f.write(" ".join(each))
#         rank_f.write("\n")

# with open("rank_eurlex4k.score", "w+") as rank_f:
#     for each in scores_list:
#         rank_f.write(" ".join(each))
#         rank_f.write("\n")
