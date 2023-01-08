import os
import pickle
from typing import List
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel,AutoTokenizer
#from scipy.spatial.distance import cosine
import torch
#from simcse import SimCSE
from utils.detector import log
from utils.premethod import read_labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_c = CrossEncoder('cross-encoder/stsb-roberta-base',device=device)
model_b = SentenceTransformer('all-MiniLM-L6-v2',device=device)
model_s = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base").to(device)
from tqdm import tqdm


def get_combine_list_sim(data_dir,pred_data,reference_data,outputdir=None)-> List[List[str]]:
    pred_data = os.path.join(data_dir,'res',pred_data)
    reference_data = os.path.join(data_dir,reference_data)
    print('pred_data: '+pred_data)
    print('reference:'+reference_data)
    outputdir=os.path.join(data_dir,'res',outputdir)
    print('data_dir: '+data_dir)
    print('write into: '+outputdir)
    print('model_name: simcse')
    all_labels= []
    pre_labels = []
    with open(reference_data,'r+') as f:
        for row in f:
            all_labels.append(row.strip())
    with open(pred_data,'r+') as f:
        for row in f:
            pre_labels.append(row.rstrip().rstrip(',').split(", "))
    
    

'''
input : 数据集目录，预测出来的标签数据，参考标签数据
output: 输出替换标签
读取all标签数据，通过simcse获得embedding，将embedding送入kmeans无监督聚类，聚类完成后获得all_label的簇组以及每个簇中心的embedding
计算预测标签的嵌入层，通过kmeans预测所属簇，然后用cross-encoder计算该标签和所有标签的的分排序，然后进行替换
'''
def cluster_combine(data_dir,pred_data,reference_data,outputdir=None)-> List[List[str]]:
    pass
#getcombine by cross-encoder
def get_combine_list(data_dir,pred_data,reference_data,outputdir=None)-> List[List[str]]:
    print('get_combine_list')
    outputdir = os.path.join(data_dir,'res',outputdir)
    print('write into: '+outputdir)
    pred_list=[]
    all_label_list=[]
    combine_list=[]
    pred_data = os.path.join(data_dir,'res',pred_data)
    reference_data = os.path.join(data_dir,reference_data)
    print('pred_data: '+pred_data)
    print('reference:'+reference_data)
    #no_in_count=[]
    #全是已经词干化的label集合
    #对应test_pregasus_pred.txt, all_stemlabels.txt
    with open(pred_data,'r+') as f1:
        for i in f1:
            pred_list.append(i.rstrip().rstrip(',').split(", "))
    with open(reference_data,'r+') as f2:
        for i in f2:
            all_label_list.append(i.rstrip())
    for i in tqdm(range(len(pred_list))):
        equal_list=[]
        no_equal_list=[]
        similar_list=[]
        for each_label in pred_list[i]:
            if each_label in all_label_list:
                equal_list.append(each_label)
            else:
                no_equal_list.append(each_label)
        #对于每一个不在已存在标签列表中的label，计算得到最相似的标签
        #no_in_count.append(len(no_equal_list[i]))
        if i%1000==0:
            print("equal:"+str(equal_list))
            print('no_equal:'+str(no_equal_list))
        for no_equal_label in no_equal_list:
            cal_score_list=[]
            for each_confer_label in all_label_list:
                cal_score_list.append((no_equal_label,each_confer_label))
            scores = model_c.predict(cal_score_list)
            scores = scores.tolist()
            #得到在参考标签中最高分的那个
            most_similar_label = all_label_list[scores.index(max(scores))]
            similar_list.append(most_similar_label)
        equal_list.extend(similar_list)
        #print("no:",no_equal_list)
        #print("similar:",similar_list)
        single_list = list(set(equal_list))
        single_list.sort(key=equal_list.index)
        combine_list.append(single_list)
    if outputdir:
        print('write into: '+data_dir+outputdir)
        with open(data_dir+outputdir,'w+')as w1:
            for row in combine_list:
                w1.write(", ".join(row)+'\n')
    # #记录替换次数
    # with open(data_dir+outputdir.rstrip('.txt')+"no_count.txt",'w+') as w1:
    #     for i in no_in_count:
    #         w1.write(str(i)+'\n')
    return combine_list

@log
def get_combine_bi_list(data_dir,pred_data,reference_data,outputdir=None)-> List[List[str]]:
    pred_data = os.path.join(data_dir,'res',pred_data)
    reference_data = os.path.join(data_dir,reference_data)
    print('pred_data: '+pred_data)
    print('reference:'+reference_data)
    outputdir=os.path.join(data_dir,'res',outputdir)
    print('data_dir: '+data_dir)
    print('write into: '+outputdir)
    pred_list=[]
    all_label_list=[]
    #no_in_count=[]
    #全是已经词干化的label集合
    #对应test_pregasus_pred.txt, all_labels.txt
    with open(pred_data,'r+') as f1:
        for i in f1:
            pred_list.append(i.rstrip().rstrip(',').split(", "))
    with open(reference_data,'r+') as f2:
        for i in f2:
            all_label_list.append(i.rstrip())
    if not os.path.exists(data_dir+"all_labels.pkl"):
        embeddings_all = model_b.encode(all_label_list,convert_to_tensor=True)
        with open(data_dir+"all_labels.pkl", "wb") as fOut:
            pickle.dump({'embeddings': embeddings_all}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_dir+'all_labels.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            #stored_sentences = stored_data['sentences']
            embeddings_all = stored_data['embeddings']
    for i in tqdm(range(len(pred_list))):
        no_equal_list=[]
        for ind,each_label in enumerate(pred_list[i]):
            #此处可以考虑不换位置，但是会更复杂，不确定是否会影响结果，单纯extend的话有可能劣化结果
            if each_label not in all_label_list:
                no_equal_list.append({'ind':ind,'label':each_label})
            #对于每一个不在已存在标签列表中的label，计算得到最相似的标签
        if len(no_equal_list)==0:
            continue
        t_list = list(map(lambda x: x['label'], no_equal_list))
        embeddings_pre = model_b.encode(t_list, convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings_pre,embeddings_all)
        #cosine_score是一个len(no_equal_list)行，(all_label_list)列的一个矩阵
        #cosine_score的长度一定等于no_equal_list
        
        flag = torch.zeros(len(all_label_list),device=device)
        for j in range(len(cosine_score)):
            this_score = torch.add(cosine_score[j],flag)
            max_ind = torch.argmax(this_score)
            #while all_label_list[max_ind] in pred_list: #if prelist has this candidate label
            #    cosine_score[j][max_ind]=0
            #    max_ind = cosine_score[j].argmax(0)
            no_equal_list[j]['label'] =  all_label_list[max_ind]
            flag[max_ind] = -2.0
        for j in no_equal_list:
            pred_list[i][j['ind']] = j['label']
    if outputdir:
        print('write into: '+outputdir)
        with open(outputdir,'w+')as w1:
            for row in pred_list:
                w1.write(", ".join(row)+'\n')

def combine_clean(data_dir,pred_data,reference_data,outputdir=None)-> List[List[str]]:              
    pred_data = os.path.join(data_dir,'res',pred_data)
    reference_data = os.path.join(data_dir,reference_data)
    print('pred_data: '+pred_data)
    print('reference:'+reference_data)
    outputdir=os.path.join(data_dir,'res',outputdir)
    print('data_dir: '+data_dir)
    print('write into: '+outputdir)
    pred_list=[]
    all_label_list=[]
    res = []
    #no_in_count=[]
    #全是已经词干化的label集合
    #对应test_pregasus_pred.txt, all_labels.txt
    with open(pred_data,'r+') as f1:
        for i in f1:
            pred_list.append(i.rstrip().rstrip(',').split(", "))
    with open(reference_data,'r+') as f2:
        for i in f2:
            all_label_list.append(i.rstrip())
    for i in range(len(pred_list)):
        tmp=[]
        for j in range(len(pred_list[i])):
            if pred_list[i][j] in all_label_list:
                tmp.append(pred_list[i][j])
        res.append(tmp)
    if outputdir:
        print('write into: '+outputdir)
        with open(outputdir,'w+')as w1:
            for row in res:
                w1.write(", ".join(row)+'\n')
    
def get_combine_simcse(data_dir,pred_data,reference_data,outputdir=None)-> List[List[str]]:
    pred_data = os.path.join(data_dir,'res',pred_data)
    reference_data = os.path.join(data_dir,reference_data)
    print('pred_data: '+pred_data)
    print('reference:'+reference_data)
    outputdir=os.path.join(data_dir,'res',outputdir)
    print('data_dir: '+data_dir)
    print('write into: '+outputdir)
    pred_list= read_labels(pred_data)
    all_label_list=[]
    #no_in_count=[]
    #全是已经词干化的label集合
    #对应test_pregasus_pred.txt, all_labels.txt
    with open(reference_data,'r+') as f2:
        for i in f2:
            all_label_list.append(i.rstrip())
    tokenizer = AutoTokenizer("princeton-nlp/sup-simcse-roberta-base")
    all_label_inputs = tokenizer(all_label_list,padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        all_label_embs = model_s(**all_label_inputs, output_hidden_states=True, return_dict=True).pooler_output
        for i in tqdm(range(len(pred_list))):
            no_equal_list=[]
            for ind,each_label in enumerate(pred_list[i]):
            #此处可以考虑不换位置，但是会更复杂，不确定是否会影响结果，单纯extend的话有可能劣化结果
                if each_label not in all_label_list:
                    no_equal_list.append({'ind':ind,'label':each_label})
            #对于每一个不在已存在标签列表中的label，计算得到最相似的标签
            if len(no_equal_list)==0:
                continue
            cur_no_equal = list(map(lambda x: x['label'], no_equal_list))
            #获取当前non-exisitent labels的inputs_ids
            cur_no_equal_token = tokenizer(cur_no_equal,padding=True, truncation=True, return_tensors="pt")
            no_equal_embs = model_s(**cur_no_equal_token, output_hidden_states=True, return_dict=True).pooler_output
            #cosine_score是一个len(cur_no_equal)行，(all_label_list)列的一个矩阵
            cos_matrix = util.cos_sim(no_equal_embs,all_label_embs)
            flag = torch.zeros(len(all_label_list),device=device)#标志向量，用于排除已经替换的参考标签
            for j in range(len(cur_no_equal)):
                cos_scores = torch.add(flag,cos_matrix[j])
                max_ind = torch.argmax(cos_scores)
                no_equal_list[j]['labels'] = all_label_list[max_ind]
                flag[max_ind] = -2.0
            for j in no_equal_list:
                #按照原来的位置替换，j['ind']是当前non-existent label的位置索引
                pred_list[i][j['ind']] = j['label']
    if outputdir:
        print('write into: '+outputdir)
        with open(outputdir,'w+')as w1:
            for row in pred_list:
                w1.write(", ".join(row)+'\n')        
                 
            
             
    
                        
        