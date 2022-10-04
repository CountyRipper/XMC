from typing import List
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import json

from tqdm import tqdm
device = 'cuda'#'cpu
def keybart_pred(model,tokenizer,document_src):
    
    ARTICLE_TO_SUMMARIZE = document_src
    #inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
  # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
    pre_result=[]
    pre_result.append(tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2))
  
    #pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])  #[2:-2]
    return str(pre_result[0])
#Original!
    """
    _summary_
    get model predicting result
    input: output_dir, src_dataname(for predicting) is document sets. 
    """
def get_pred_Keybart(dir,output_dir,src_dataname,model_path):
    '''
    加载配置
    dir是当前数据库文件夹位置
    outputdir是输出预测文件位置： dir+generate_result
    srcdata是用于预测的原始文件doc的位置 dir+
    '''
    output_dir = dir+output_dir
    print('output: '+output_dir)
    src_dataname = dir+src_dataname
    print('src_data: '+src_dataname)
    model_path = dir+model_path
    print("model_path: "+model_path)
    # model save dir
    #dir = './dataset/EUR-Lex/ dataset dir
    #model path = "./keybart_save"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    #model = 
    #model = PegasusForConditionalGeneration.from_pretrained(model_path).to(device)#BART-large-Finetuned
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #tokenizer = PegasusTokenizer.from_pretrained(model_path)

    tokenizer.save_pretrained(dir+"keybart_tokenizer")
    tokenizer.save_vocabulary(dir+"keybart_tokenizer")
    #print(tokenizer.vocab_size)
    tokenizer.get_added_vocab()
    data = []
    dic = [] # dictionary for save each model generate result
    src_value = [] # using for get source document which is used to feed into model, and get predicting result
    res = []
    # open test file 
    with open(src_dataname, 'r+') as f:
        for line in f:
            data.append(json.loads(line))
        # 进度条可视化 vision process
        for i in tqdm(range(len(data))): #range(len(data))
            dic = data[i]
            src_value = dic["document"]
            tmp_result = keybart_pred(model,tokenizer,src_value)
            #print(tmp_result)
            # pre_labels = tmp_result.strip('"').strip("[]").split(",")
            # l = list(map( lambda x: x.strip().strip("'"),pre_labels))
            # dic["pred"] = lg
            # sign= ", "
            # res.append(sign.join(l))
            # if i%1000==0:
            #     print(res[i])
            dic["pred"] = tmp_result.replace ('\\','')
            res_labels=[]
            pre_result=dic["pred"].strip("'").strip("[]").strip('\"').strip("[]").strip("\\").strip("'").split(",")
            for j in range(len(pre_result)):
                tmpstr = pre_result[j].strip(" ").strip("'")
                if tmpstr=='':
                    continue
                res_labels.append(tmpstr)
            # if output_dir:
            #     with open(output_dir,'a+') as t:
            #         #json.dump(dic,t)
            #         t.write(res[i])
            #         t.write('\n')
            sign= ", "
            res.append(sign.join(res_labels))
            if i%1000==0:
                print(res[i])
            with open(output_dir,'a+') as t:
                #json.dump(dic,t)
                t.write(res[i])
                t.write('\n')

def zero_shot_keybart_generation(dir,output_dir,src_dataname) ->List[str]:
    output_dir = dir+output_dir
    print('output: '+output_dir)
    src_dataname = dir+src_dataname
    print('src_data: '+src_dataname)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
    data=[]
    res: List[str]=[]
    with open(src_dataname, 'r+') as f:
        for line in f:
            data.append(json.loads(line))
        for i in tqdm(range(len(data))): #range(len(data))
            dic = data[i]
            doc_value = dic["document"]
            tmp_result = keybart_pred(model,tokenizer,doc_value)
            print(tmp_result)
            res.append(tmp_result)
    return res
    