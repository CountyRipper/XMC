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
    pass
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
    """
def stem_labels(data_name,outputname=None)->List[str]:
    # pay more attention, data_name should be merged
    with open(data_name+".txt","r+") as label_set:
        stemed_result_list=[] # to save stemed word set
        for line in label_set:
            cur_label_set = line.split(" ")# pay attention to format
            stem_result=[]
            for per_label in cur_label_set:
                word_list=[] # to save stemed words
                for per_word in per_label.split("_"):
                    word_list.append(stemmer2.stem(per_word))
                stem_result.append(" ".join(word_list))
            stemed_result_list.append(stem_result)
        if outputname:
            with open(outputname+".txt","w+") as f:
                for i in stemed_result_list:
                    f.write(i+"\n")
        return stemed_result_list