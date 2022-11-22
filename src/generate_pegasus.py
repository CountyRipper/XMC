from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
import json
from premethod import batch_pred
# import torch
# import ast
from tqdm import tqdm
from torch.utils.data import DataLoader
device = 'cuda'#'cpu
def pegasus_pred(model,tokenizer,model_path,src):
    
    ARTICLE_TO_SUMMARIZE = src
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
  # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length = 64,num_beams = 8).to(device)  #length_penalty = 3.0  top_k = 5
    pegasus_pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])#[2:-2]
    return pegasus_pred
def pegasus_pred_fast(model,tokenizer,src):
    ARTICLE_TO_SUMMARIZE = src
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
  # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length = 64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
    #pegasus_pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])#[2:-2]
    pre_result=[]
    for g  in summary_ids:
        pre_result.append(tokenizer.decode(g,skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return pre_result[0]
#Original!
    """
    _summary_
    get model predicting result
    input: output_dir, src_dataname(for predicting) is document sets. 
    """
def get_pred_Pegasus(dir,output_dir,src_dataname,model_path):
    '''
    加载配置
    dir是当前数据库文件夹位置
    outputdir是输出预测文件位置： dir+generate_result
    srcdata是用于预测的原始文件doc的位置 dir+
    '''
    output_dir = dir+output_dir
    model_path = dir+model_path
    src_dataname = dir+src_dataname
    print('src_data: '+src_dataname)
    print("model_path: "+model_path)
    print("output: "+output_dir)
    # model save dir
    #dir = './dataset/EUR-Lex/ dataset dir
    model = PegasusForConditionalGeneration.from_pretrained(model_path).to(device)#BART-large-Finetuned
    tokenizer = PegasusTokenizerFast.from_pretrained(model_path)

    tokenizer.save_pretrained(dir+"pegasus_tokenizer")
    tokenizer.save_vocabulary(dir+"pegasus_tokenizer")
    print(tokenizer.vocab_size)
    tokenizer.get_added_vocab()
    
    data = []
    dic = [] # dictionary for save each model generate result
    src_value = [] # using for get source document which is used to feed into model, and get predicting result
    pre_result = [] #get model predicting result. (each points data)
    res = []
    # open test file 
    with open(src_dataname, 'r+') as f:
        for line in f:
            data.append(json.loads(line))
        # 进度条可视化 vision process
        for i in tqdm(range(len(data))): #range(len(data))
            dic = data[i]
            src_value = dic["document"]
            tmp_result = pegasus_pred(model,tokenizer,model_path,src_value)
            #print(tmp_result)
            #tmp_result=tmp_result.strip('[').strip(']').strip('"').strip('[').strip(']')
            #tmp_result=tmp_result.split(',')
            #print(tmp_result)
            dic["pred"] = tmp_result.replace ('\\','')
            #del dic['id']
            #del dic['document']
            #del dic['summary']
            #print(dic)
            #print(dic["pred"])
            res_labels=[]
            pre_result=dic["pred"].strip("'").strip("[]").strip('\"').strip("[]").strip("\\").strip("'").split(",")
            for j in range(len(pre_result)):
                tmpstr = pre_result[j].strip(" ").strip("'")
                if tmpstr=='':
                    continue
                res_labels.append(tmpstr)
            #pre_result = list(map(lambda x : str(x).strip("\'").strip("\'"), pre_result))#python3 map() return an iteration
            # write result set into output file
            #把这个步骤外移，
            sign= ", "
            res.append(sign.join(res_labels))
            if i%1000==0:
                print(res[i])
            with open(output_dir,'a+') as t:
                #json.dump(dic,t)
                t.write(res[i])
                t.write('\n')
    #print('pred ending, out range pred nums:'+str(outrangenum))
        # with open(dir+output_dir+"test_pegasus_pred.txt",'w+') as w:
        #     for row in res:
        #         w.write(row+"\n")
        #f.close()
        #t.close()
#get_pred_Pegasus("generate_result","test_finetune.json","pegasus_test_save")
def get_pred_Pegasus_fast(dir,output_dir,src_dataname,model_path):
    output_dir = dir+output_dir
    model_path = dir+model_path
    src_dataname = dir+src_dataname
    print('src_data: '+src_dataname)
    print("model_path: "+model_path)
    print("output: "+output_dir)
    # model save dir
    #dir = './dataset/EUR-Lex/ dataset dir
    model = PegasusForConditionalGeneration.from_pretrained(model_path).to(device)#BART-large-Finetuned
    tokenizer = PegasusTokenizerFast.from_pretrained(model_path)
    print('this is the check'+'\n'+'******************************************'+"\n")
    tokenizer.save_pretrained(dir+"pegasus_tokenizer")
    tokenizer.save_vocabulary(dir+"pegasus_tokenizer")
    print(tokenizer.vocab_size)
    tokenizer.get_added_vocab()
    print('this is the check2'+'\n'+'******************************************'+"\n")
    
    data = []
    dic = [] # dictionary for save each model generate result
    src_value = [] # using for get source document which is used to feed into model, and get predicting result
    res = []
    batch=[]
    # open test file 
    with open(src_dataname, 'r+') as f:
        for line in f:
            data.append(json.loads(line)['document'])
        dataloader = DataLoader(data,batch_size=16)
        # 进度条可视化 vision process
        f=open(output_dir,'w+')
        f.close()
        with open(output_dir,'a+') as t:
           for i in tqdm(dataloader): #range(len(data))
                batch = i
                tmp_result = batch_pred(model,tokenizer,batch)
                for j in tmp_result:
                    l_labels = [] #l_label 是str转 label的集合
                    pre = j.strip('[]').strip().split(",")
                    for k in range(len(pre)):
                        tmpstr = pre[k].strip(" ").strip("'").strip('"')
                        if tmpstr=='':continue
                        l_labels.append(tmpstr)
                    res.append(l_labels)
                    t.write(", ".join(l_labels))
                    t.write("\n")
            #for i in res:
                #t.write(res)
    return res