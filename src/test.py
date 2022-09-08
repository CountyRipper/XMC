# list=  [['a','aa','aaa'],['b','bbb','ab']]
# with open('test.txt','w+') as w:
#     for i in list:
#         tmpstr=", "
#         tmpstr = tmpstr.join(i)
#         w.write(tmpstr+"\n")
#datadir='./dataset/EUR-Lex/'
# list1=[('a',3),('b',1),('c',2)]
# print(list1)
# list1.sort(key= lambda x:x[1])
# print(list1)
# list2 = list(map(lambda x: x[1],list1))
# print(list2)
from keybart_generate import *
from transformers import pipeline
datadir = ['./dataset/EUR-Lex/','./dataset/Wiki500K/']
tasks = ['test','train','valid']
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
doc:List[str] = []
res:List[str]=[]
with open(datadir[0]+tasks[0]+"_finetune.json",'r') as f:
    for row in f:
        doc.append(json.loads(row)['document'])
for i in range(len(doc)):
    res.append(summarizer(doc[i], max_length=130, min_length=30, do_sample=False))        
print(res)

#list1 = get_pred_Keybart(datadir[0],"generate_result/"+tasks[0]+"_kb_test_pred.txt",tasks[0]+"_finetune.json","keybart_save")
#list = zero_shot_keybart_generation(datadir[0],"generate_result/"+tasks[0]+"_kb_pred.txt",tasks[0]+"_finetune.json")


        
            