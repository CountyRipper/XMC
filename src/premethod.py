from logging import Logger
import json
import logging
import os
from typing import List
from tqdm import tqdm
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from utils.detector import log
#from xclib.data import data_utils  
nltk.download('stopwords')
#stemmer = SnowballStemmer("english")
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


'''
get all labels without repulit.
input: franc, beef, intervent stock, aid to disadvantag group,
        franc, beef,communiti aid,
output: franc \n beef \n intervent stock \n
'''
def get_all_labels(data_name,outputname=None)-> List[str]:
    #约定dataname第一个是train，第二个是test，第三个valid
    label_set=[]
    for i in data_name:
        with open(i,'r+') as file:
            for row in file:
                label_set.append(row.rstrip())
    with open(outputname, 'w+') as res:
        for row in label_set:
            for j in row.split(', '):
                res.write(j+"\n")
@log                
def get_all_labels_allfile(datas,outputname=None)-> List[str]:
    #获取多个文件的所有labels，并且去重
    label_set=set()
    for i in datas:
        with open(i,'r+') as f:
            for row in f:
                for each in row.strip('\n').split(", "):
                    label_set.add(each)
    with open(outputname,'w+') as w:
        for j in list(label_set):
            w.write(j+"\n")
        
    #return label_set
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
    如果已经是aid_to_disadvantag_group这种合并类型，不要再使用merge_label
    """
def stem_labels(data_name,outputname=None)->List[str]:
    # pay more attention, data_name should be merged
    stem_result=[] # two dimension set. each record-> one stemed word sets;
    with open(data_name+".txt","r+") as label_set:
        for row in tqdm(label_set):
            row =row.strip()
            word_array = row.split(" ") # each unstemed words array in each row
            stem_label_result=[]
            for each_label in word_array:
                word_list = []  #存储每一个标签切分之后的临时数组
                for each_word in each_label.split("_"): #词组形式用“_”切分
                    word_list.append(stemmer2.stem(each_word))
                stem_label_result.append(" ".join(word_list)) #将每一个词干化好的标签添加到结果数组
            stem_result.append(", ".join(stem_label_result))   
        if outputname: #如果需要输出文件
            with open(outputname+".txt","w", encoding='utf8') as f:
                for i in stem_result:
                    f.write(str(i)+"\n")
        return stem_labels  
   
"""
将txt转化为json来训练
outputname shold be: data_name+".json",注意output dir，并且都是词干化的
"""
def txt_to_json(text_name,label_name,outputname=None):
    text_path = text_name+".txt"
    label_path = label_name+".txt"
    json_path = outputname+".json"
    pair={} #定义字典，用于组合相关元素
    with open(json_path,'w+') as w:
        all_text_data = [] #所有行数据（总数据）
        with open(text_path,'r+') as t:
            all_label_data = []
            with open(label_path,'r+') as l:
                for row in t:
                    all_text_data.append(row)
                for row in l:
                    all_label_data.append(row.rstrip().split(", "))
                for i in tqdm(range(len(all_text_data))):
                    pair['document'] = all_text_data[i].rstrip()
                    pair['id'] = i
                    #请注意接下来无词干化过程，需要确保词干化的label文件
                    #print(all_label_data[i])
                    pair['summary'] = str(all_label_data[i])
                    if i %10000 == 0:
                        print(str(i)+": "+ str(len(all_text_data)))
                        print(str(all_label_data[i]))
                    json.dump(pair,w)
                    w.write("\n")
'''
将json转化为对应的text和label的文件
'''
def json_to_text(file_name):
    json_file = file_name+"_finetune.json"
    text_name = file_name+"_texts.txt"
    label_name = file_name+"_labels.txt"
    text = []
    label = []
    js=[]
    with open(json_file,'r+') as f:
        for row in f:
            js.append(json.loads(row))
        for i in js:
            text.append(i['document'])
            label.append(i['summary'])
        with open(text_name,'w+')as w:
            for j in text:
                w.write(str(j)+'\n')
        with open(label_name,'w+') as w:
            for j in label:
                w.write(", ".join(eval(j))+'\n')
                
'''
get all label_stem
获取全部词干化的单词
'''
def get_all_stemlabels(datadir,outputdir=None)-> List[str]:
    #理论上应该是test+train+valid所有的stem_labels文件合并在一起
    print('get_all_stemlabels:begin()')
    res = set()
    with open(datadir,'r+') as file:
        for i in tqdm(file):
            # row 是标签
            res.add(i)
    if outputdir:
        with open(outputdir,'w+') as w:
            for i in res:
                w.write(str(i))
    print('get_all_stemlabels:end()')
    print('all_label_stem outputdir: '+ outputdir )
    return res

        
def bart_clean(data_name,outputname=None)-> List[str]:
    #input dir = XXX_pred.txt?
    '''
    替换思路： “ 全部替换为空， [' ']全部替换为空，仅限eurlex-4k
    '''
    logger.info("bart_clean")
    logger.info("data_name: "+data_name)
    logger.info("outputname: "+outputname)
    res = []
    with open(data_name,'r+') as f:
        for row in f:
            tmp_list = row.split(", ")
            label_list=[]
            #bug_list=[['- ','-'],[' -','-'],["' ","'"],[" '","'"],['" ','"'],[' "','"']]
            
            label_list = list(map(lambda x: x.replace('["',", ").replace('"]',", ").replace("['",", ").replace("']",", ").replace('"','').replace("' ",", ").replace(" '",","),tmp_list))
            #label_list = list(map(lambda x:x))  
            #flag=True
            bracelet={"l":0,"r":0}
            for i in range(len(label_list)):
                for j in range(len(label_list[i])):
                    if label_list[i][j]=="(": bracelet["l"]=1
                    if label_list[i][j]==")": bracelet["r"]=1
                if bracelet["l"]==1 and bracelet["r"]==1:
                    i=i
                else: 
                    label_list[i]=label_list[i].replace("(","").replace(")","")
                                                
            res.append(label_list)                      
    if outputname:
        with open(outputname,"w+") as w:
            for row in res:
                w.write(", ".join(row))
                    
                

def single_pred(model,tokenizer,document_src):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #ARTICLE_TO_SUMMARIZE = document_src
    inputs = tokenizer([document_src], return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
  # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
    pre_result=[]
    for g  in summary_ids:
        pre_result.append(tokenizer.decode(g,skip_special_tokens=True, clean_up_tokenization_spaces=True))
    #pegasus_pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]  #[2:-2]
    return pre_result[0]

def batch_pred(model,tokenizer,documents)->List[List]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #if documents==[]:return
    #ARTICLE_TO_SUMMARIZE = document_src
    #加速，集中处理
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)
    #inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
  # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =48,num_beams = 5).to(device)
    #summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
    pre_result=tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2)
    #pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])  #[2:-2]
    return pre_result
            
def xml_para(src_data,outputdata=None):
    tree = ET.parse(src_data)
    root = tree.getroot()
    res = []
    for each in root:
        for i in each:
            if(i.tag=='tags'):
                for j in i:
                    for k in j :
                        if k.tag=='name':
                            res.append(k.text)
    if outputdata:
        with open(outputdata,'w+')as w:
            for i in res:
                w.write(i+"\n")        
    return res

# def bow_label_map(text_data,f_data,output=None):
#     # Read file with features and labels (old format from XMLRepo) 'train.txt'
#     features, tabels, num_samples, num_features, num_labels = data_utils.read_data(text_data)
#     # Read sparse file (see docstring for more) f_data='trn_X_Xf.txt'
#     # header can be set to false (if required)
#     mylabels = []
#     mylabels = data_utils.read_sparse_file(f_data, header=True)

# # Write sparse file (with header)
#     data_utils.write_sparse_file(mylabels, output)
    
def k_fold_split(dir,outputdir,k:int = 5):
    #输入数据文件，输出目标文件夹，输入包括train/test_finetune.json
    #train/test_texts.txt   train/test_labels_stem.txt(词干化)
    #分割出k个
    print('k_fold_split')
    print('dir: '+dir)
    print('output: '+ outputdir)
    train_data = dir+"train_finetune.json"
    test_data = dir+"test_finetune.json"
    train_text = dir+"train_texts.txt"
    train_labels = dir+"train_labels_stem.txt"
    test_text = dir+"test_texts.txt"
    test_labels = dir+"test_labels_stem.txt"
    label_sum = []
    text_sum = []
    sum = []
    with open(train_data,'r+') as tf:
        for row in tf:
            sum.append(json.loads(row))
    with open(test_data,'r+') as tf:
        for row in tf:
            sum.append(json.loads(row))
    with open(train_text,'r+') as f:
        for row in f:
            text_sum.append(row)
    with open(test_text,'r+') as f:
        for row in f:
            text_sum.append(row)
    with open(train_labels,'r+') as f:
        for row in f:
            label_sum.append(row)
    with open(test_labels,'r+') as f:
        for row in f:
            label_sum.append(row)
    for i in range(k):
        test_j=[]
        train_j=[]
        test_l=[]
        train_l=[]
        test_t=[]
        train_t=[]
        for j in range(len(sum)):
            if j%5==i:
                test_j.append(sum[j])
                test_l.append(label_sum[j])
                test_t.append(text_sum[j])
            else:
                train_j.append(sum[j])
                train_l.append(label_sum[j])
                train_t.append(text_sum[j]) 
        cur_dir= outputdir+"K_"+str(i)+"/"       
        with open(cur_dir+"test.json",'w+') as w:
            for i in test_j:
                json.dump(i,w)
                w.write("\n")
        with open(cur_dir+"train.json",'w+') as w:
            for i in train_j:
                json.dump(i,w)
                w.write("\n")
        with open(cur_dir+"train_texts.txt",'w+') as w:
            for i in train_t:
                w.write(i)
        with open(cur_dir+"test_texts.txt",'w+') as w:
            for i in test_t:
                w.write(i)
        with open(cur_dir+"train_labels_stem.txt",'w+') as w:
            for i in train_l:
                w.write(i)
        with open(cur_dir+"test_labels_stem.txt",'w+') as w:
            for i in test_l:
                w.write(i)
@log
def split_jsonfile(data_dir,output_dir,k =10):
    # 切分 json文件
    logger.info("split_file:")
    logger.info("data_dir: "+data_dir)
    logger.info("output_dir: "+output_dir)
    sum=[]
    with open(data_dir,'r+') as f:
        for ind, row in enumerate(tqdm(f)):
            if ind%k==0:
                sum.append(json.loads(row))
    with open(output_dir,'w+') as w:
        for j in tqdm(sum):
            json.dump(j,w)
            w.write("\n")
    logger.info("split "+data_dir+" end.")
def wiki500_change(json_dir,map_dir,js_output_dir,label_output_dir,texts_output_dir):
    #拿取text和标签index，然后用label_dir替换标签
    logger.info("wiki500_change")
    logger.info("json_dir: "+json_dir)
    logger.info("map_dir:"+map_dir)
    logger.info("js_output_dir: "+js_output_dir)
    logger.info("label_output:"+label_output_dir)
    logger.info("texts_output_dir"+texts_output_dir)
    doc=[]
    label_map = []
    with open(map_dir,'r+') as map_f:
        for row in tqdm(map_f):
            label = row.replace("Category:","").split("->")[0]
            label = label.replace("_"," ")
            label_map.append(label)
    #js_dic={}
    with open(json_dir,'r+') as j_file:
        for i in tqdm(j_file):
            js_dic={}
            js_f = json.loads(i)
            #仅取content和target_ind
            js_dic['document'] = js_f['content']
            js_dic['target_ind'] = js_f['target_ind']
            doc.append(js_dic)
    label_set=[]
    for i in range(len(doc)):
        label_list = []
        for j in doc[i]['target_ind']:
            label_list.append(label_map[j].replace("_"," "))
        label_set.append(label_list)
        doc[i]['summary']= str(label_list)
        #del doc[i]['target_ind']
       
    with open(js_output_dir,'w+') as w:
        for j in tqdm(doc):
            json.dump(j,w)
            w.write("\n")
    with open(label_output_dir,'w+') as w:
        for j in label_set:
            sign = ", "
            w.write(sign.join(j))
            w.write("\n")
    with open(texts_output_dir,'w+') as w:
        for j in tqdm(doc):
            w.write(j['document'])
            w.write("\n")
    
def amazoncat_change(json_dir,map_dir,js_output_dir,label_output_dir,texts_output_dir):
    #拿取text和标签index，然后用label_dir替换标签
    logger.info("json_dir: "+json_dir)
    logger.info("map_dir:"+map_dir)
    logger.info("js_output_dir: "+js_output_dir)
    logger.info("label_output:"+label_output_dir)
    logger.info("texts_output_dir"+texts_output_dir)
    doc=[]
    label_map = []
    map_f = open(map_dir,'r+',encoding='ISO-8859-1')
    for row in tqdm(map_f):
        label_map.append(row.strip('\n'))
    map_f.close()
    # with open(map_dir,'r+', encoding='ISO-8859-15') as map_f:
    #     for row in map_f:
    #         label_map.append(str(row).split("\n"))

    with open(json_dir,'r+') as j_file:
        for i in tqdm(j_file):
            js_dic={}
            js_f = json.loads(i)
            #仅取content和target_ind
            js_dic['document'] = 'title: '+js_f['title'].strip('\n')+", "+"content: "+js_f['content']
            js_dic['target_ind'] = js_f['target_ind']
            doc.append(js_dic)
    label_set=[]
    for i in range(len(doc)):
        label_list = []
        for j in doc[i]['target_ind']:
            label_list.append(label_map[j])
        label_set.append(label_list)
        doc[i]['summary']= str(label_list)
        #del doc[i]['target_ind']       
    with open(js_output_dir,'w+') as w:
        for j in tqdm(doc):
            json.dump(j,w)
            w.write("\n")
    with open(label_output_dir,'w+') as w:
        for j in tqdm(label_set):
            sign = ", "
            w.write(sign.join(j))
            w.write("\n")
    with open(texts_output_dir,'w+') as w:
        for j in tqdm(doc):
            w.write(j['document'])
            w.write("\n")
           
                
def clean_b(data_name):
    #clea_row=[]
    clean_set=[] #save clean_row
    with open(data_name+".txt","r") as raw_text:
        for row in raw_text:
            row = row.strip("[").strip("]").strip("\"")
            print(row)
            #clea_row=row
            clean_set.append(row)
        with open(data_name+"_b"+".txt","w+") as clean_text:
            for i in clean_set:
                clean_text.write(i)
            clean_text.write("\n")
# dataname="test_pegasus_combine_stem"
# clean_b(dataname)        

def clean_set(datadir,filepath):
    filepath = os.path.join(datadir,filepath)
    res = []
    with open(filepath, 'r+') as f:
        for row in f:
            label_list = row.strip('\n').split(", ")
            s_labels = set()
            for i in label_list:                
                s_labels.add(i)
            res.append(list(s_labels))
    with open(filepath,'w+') as w:
        for i in res:
            w.write(", ".join(i)+"\n")
            
    