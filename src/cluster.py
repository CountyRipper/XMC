
from numpy import float32
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np
from detector import log
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@log
def get_embedding(datadir,srcdata):
    print(device)
    #获取all——labels的嵌入向量，用于k-means
    print("datadir: "+datadir)
    srcdata = datadir+srcdata
    print("srcdir: "+srcdata)
    # Import our models. The package will take care of downloading the models automatically
    #princeton-nlp/sup-simcse-roberta-large
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").to(device)
    texts = [] #all labels
    with open(srcdata,'r+') as f:
        for row in f:
            texts.append(row.strip('\n'))
    res = []
    # Tokenize input texts
    batch=[]
    for i in tqdm(range(len(texts))):
        if (i==0 or i%1000!=0) and i<len(texts)-1:
            batch.append(texts[i])
        else:
            batch.append(texts[i])
            #batch=torch.Tensor(batch).cuda()
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                for j in embeddings:
                    res.append(j)
            batch=[]
    return res

def get_means(embedding_list:torch.tensor,output=None):
    #获得每个label对应的簇属，返回对应的字典列表
    nplist = []
    for i in embedding_list:
        nplist.append(i.cpu().numpy())
    X = np.array(nplist,dtype='float64')
    km = KMeans(n_clusters=8,random_state=0,max_iter=100)
    #km.n_iter_ = 10000
    km.fit(X)
    res=[]
    clusters = km.cluster_centers_
    for ind,i in enumerate(nplist):
        res.append({'k_index':km.predict([i]),'src':ind})
    if output:
        with open(output,'w+') as w:
            for i in res:
                w.write(str(i)+"\n")
    # Get the embeddings
    