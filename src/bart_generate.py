from sre_parse import Tokenizer
from typing import List
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from premethod import single_pred
from tqdm import tqdm
device = 'cuda'#'cpu


def get_pred_bart(dir,output_dir,src_dataname,model_path):
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
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    #tokenizer = PegasusTokenizer.from_pretrained(model_path)

    tokenizer.save_pretrained(dir+"bart_tokenizer")
    tokenizer.save_vocabulary(dir+"bart_tokenizer")
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
            tmp_result = single_pred(model,tokenizer,src_value)
            dic["pred"] = tmp_result.replace ('\\','')
            res_labels=[]
            pre_result=dic["pred"].strip("'").strip("[]").strip('\"').strip("[]").strip("\\").strip("'").split(",")
            for j in range(len(pre_result)):
                tmpstr = pre_result[j].strip(" ").strip("'")
                if tmpstr=='':
                    continue
                res_labels.append(tmpstr)
            sign= ", "
            res.append(sign.join(res_labels))
            if i%1000==0:
                print(res[i])
            with open(output_dir,'a+') as t:
                #json.dump(dic,t)
                t.write(res[i])
                t.write('\n')
    return res
    