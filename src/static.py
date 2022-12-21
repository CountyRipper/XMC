import os
from typing import List
import pke
from utils.p_at_1 import p_at_k
from tqdm import tqdm
def get_candidate(method:str,num:int,text_dir,output_dir=None)->List[str]:
    model = None
    texts = []
    ress=[]
    resso=[]
    if 'toprank' in method:
        model = pke.unsupervised.TopicRank()
    elif 'tfidf' in method:
        model = pke.unsupervised.TfIdf()
    else:
        model = pke.unsupervised.TfIdf()
    with open(text_dir,'r+')as f:
        for i in f:
            texts.append(i.strip())
    for each_text in tqdm(texts):
        model.load_document(input=each_text, language='en')
        #model.grammar_selection(grammar=grammar)
        model.candidate_selection()
        model.candidate_weighting()  
        res = model.get_n_best(n=num, stemming=True)
        ress.append([i[0] for i in res])
        resso.append([i[1] for i in res])
    if output_dir:
        with open(output_dir,'w+') as w:
            for i in ress:
                w.write(", ".join(i))
                w.write('\n')
    return ress  
dir = "./dataset/EUR-Lex/"
get_candidate("toprank",5,dir+"test_texts.txt",os.path.join(dir,'res','test_toprank.txt'))
get_candidate("tfidf",5,dir+"test_texts.txt",os.path.join(dir,'res','test_tfidf.txt'))
p_at_k(dir,"test_labels.txt",'test_toprank.txt','res_toprank.txt')
p_at_k(dir,"test_labels.txt",'test_tfidf.txt','res_tfidf.txt')
