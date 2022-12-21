from keybert import KeyBERT
import os
from utils.p_at_1 import p_at_k
from tqdm import tqdm

def get_candidate_kw(modelname,min_num,max_num,text_dir,output_dir=None):
    texts = []
    keys = []
    scores = []
    with open(text_dir,'r+')as f:
        for i in f:
            texts.append(i.strip())
    kw_model = KeyBERT(model=modelname)
    for i in tqdm(texts):
        res = kw_model.extract_keywords(i, keyphrase_ngram_range=(min_num, max_num), stop_words=None)
        keys.append([i[0] for i in res])
        scores.append([i[1] for i in res])
    if output_dir:
        with open(output_dir,'w+') as w:
            for i in keys:
                w.write(", ".join(i))
                w.write('\n')
    return keys
#dir = "./dataset/Wiki10-31K/"
#get_candidate_kw('all-MiniLM-L6-v2',1,1,dir+"test_texts.txt",os.path.join(dir,'res','test_kw.txt'))
#p_at_k(dir,"test_labels.txt",'test_kw.txt','res_kw.txt')
dir = "./dataset/Wiki500K-20/"
get_candidate_kw('all-MiniLM-L6-v2',1,1,dir+"test_texts.txt",os.path.join(dir,'res','test_kw.txt'))
p_at_k(dir,"test_labels.txt",'test_kw.txt','res_kw.txt')
dir = "./dataset/Wiki500K/"
get_candidate_kw('all-MiniLM-L6-v2',1,1,dir+"test_texts.txt",os.path.join(dir,'res','test_kw.txt'))
p_at_k(dir,"test_labels.txt",'test_kw.txt','res_kw.txt')