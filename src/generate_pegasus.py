from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json
import torch
import ast
from tqdm import tqdm


device = 'cuda'#'cpu
# model save dir
model = PegasusForConditionalGeneration.from_pretrained('./pegasus_test_save').to(device)#BART-large-Finetuned
tokenizer = PegasusTokenizer.from_pretrained('./pegasus_test_save')

tokenizer.save_pretrained("added")
tokenizer.save_vocabulary("added")
print(tokenizer.vocab_size)
tokenizer.get_added_vocab()

def pegasus_pred(src):
    ARTICLE_TO_SUMMARIZE = src
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True


  # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length = 64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
    pegasus_pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])#[2:-2]
    return (pegasus_pred)

#Original!
    """_summary_
    get model predicting result
    input: output_dir, src_dataname(for predicting) is document sets. 
    """
def get_pred_Pegasus(output_dir,src_dataname):
    data = []
    dic = [] # dictionary for save each model generate result
    src_value = [] # using for get source document which is used to feed into model, and get predicting result
    pre_result = [] #get model predicting result. (each points data)
    # open test file 
    with open("./dataset/"+src_dataname, 'r+') as f:
        for line in f:
            data.append(json.loads(line))
        # 进度条可视化 vision process
        for i in tqdm(range(len(data))): #range(len(data))
            dic = data[i]
            src_value = dic["document"]
            tmp_result = pegasus_pred(src_value)
            #print(tmp_result)
            #tmp_result=tmp_result.strip('[').strip(']').strip('"').strip('[').strip(']')
            #tmp_result=tmp_result.split(',')
            #print(tmp_result)
            dic["pred"] = tmp_result.replace ('\\','')
            #del dic['id']
            #del dic['document']
            #del dic['summary']
            #print(dic)
            pre_result=dic["pred"].strip("[]").strip('\"').strip("'").strip("[]").strip(" ").split(",")
            for i in range(len(pre_result)):
                tmpstr = pre_result[i].split("'")
                pre_result[i] = tmpstr[1]
            #pre_result = list(map(lambda x : str(x).strip("\'").strip("\'"), pre_result))#python3 map() return an iteration
            # write result set into output file
            with open("./"+output_dir+"/"+"test_pegasus_pred.txt",'a+') as t:
                #json.dump(dic,t)
                for i in pre_result:
                    t.write(i+", ")
                t.write('\n')
        #f.close()
        #t.close()
get_pred_Pegasus("generate_result","test_finetune.json")
