import json
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
'''
本py文件用于数据预处理，将txt转化为json数据结构便于处理
json template:
{"document": "commiss decis juli lai detail "id": 0, "summary": "['award of contract', 'aid to disadvantag group']"}
'''

pair = {}
#dataset_type = ["train", "valid", "test"]
#dataset_type = ["train", "test"]
dataset_type = ["train", "test"]

def for_finetune(dataset_name):
    data_dir="../dataset/"
    finetuned_path = data_dir+dataset_name + "_finetune.json" #finetune_path is 
    # text是src，label是tgt 
    text_path = data_dir+dataset_name + "_texts.txt"
    label_path = data_dir+dataset_name + "_labels.txt"

    with open(finetuned_path,'w+') as w: #写模式打开finetuned_path文件
        text_data = []
        with open(text_path, 'r+') as s: #读模式打开text_path
            label_data = []
            with open(label_path, 'r+') as t: #读模式打开src_Path
                for line in t:
                    label_data.append(line)
                for line in s:
                    text_data.append(line)
                for i in range(len(text_data)):
                    pair["document"] = text_data[i].rstrip()
                    pair["id"] = i
                    summary_list = label_data[i].rstrip().split(" ")
                    summary = []
                    #词干化--过耦合（待修改）应该直接使用词干化并去_的labels文件
                    for each in summary_list:
                        split_one = each.split("_")
                        one_label = [stemmer2.stem(plural) for plural in split_one]
                        summary.append(" ".join(one_label))
                    pair["summary"] = str(summary)
                    #result = json.dumps(eval(str(pair)))
                    if i % 10000 == 0:
                        print(str(i) + " / " + str(len(text_data)))
                        print(str(summary))
                    json.dump(pair,w)
                    w.write('\n')

for each in dataset_type:
    for_finetune(each)
