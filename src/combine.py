from typing import List
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/stsb-roberta-base')
from rouge import Rouge
from tqdm import tqdm

#getcombine by cross-encoder
def get_combine_list(data_dir,pred_data,reference_data,outputdir=None)-> List[List[str]]:
    print('get_combine_list')
    print('write into: '+data_dir+outputdir)
    pred_list=[]
    all_label_list=[]
    combine_list=[]
    no_in_count=[]
    #全是已经词干化的label集合
    #对应test_pregasus_pred.txt, all_stemlabels.txt
    with open(data_dir+pred_data,'r+') as f1:
        for i in f1:
            pred_list.append(i.rstrip().rstrip(',').split(", "))
    with open(data_dir+reference_data,'r+') as f2:
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
        no_in_count.append(len(no_equal_list[i]))
        if i%1000==0:
            print("equal:"+str(equal_list))
            print('no_equal:'+str(no_equal_list))
        for no_equal_label in no_equal_list:
            cal_score_list=[]
            for each_confer_label in all_label_list:
                cal_score_list.append((no_equal_label,each_confer_label))
            scores = model.predict(cal_score_list)
            scores = scores.tolist()
            #得到在参考标签中最高分的那个
            most_similar_label = all_label_list[scores.index(max(scores))]
            similar_list.append(most_similar_label)
        equal_list.extend(similar_list)
        #print("no:",no_equal_list)
        #print("similar:",similar_list)
        combine_list.append(equal_list)
    if outputdir:
        print('write into: '+data_dir+outputdir)
        with open(data_dir+outputdir,'w+')as w1:
            for row in combine_list:
                w1.write(", ".join(row)+'\n')
    with open(data_dir+outputdir.strip('.txt')+"nn_count.txt",'w+') as w1:
        for i in no_in_count:
            w1.write(str(i)+'\n')
    return combine_list
                        
        