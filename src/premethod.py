from base64 import encode
from head import *
 
nltk.download('stopwords')
#stemmer = SnowballStemmer("english")
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)

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
        
        
    