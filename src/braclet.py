# "["beef common organi of market"]"去除括号
def clean_b(data_name):
    #clea_row=[]
    clean_set=[] #save clean_row
    with open(data_name+".txt","r") as raw_text:
        for row in raw_text:
            row = row.strip("[").strip("]").strip("\"")
            print(row)
            #clea_row=row
            clean_set.append(row)
        with open(data_name+"_b"+".txt","w+") as clean_text:
            for i in clean_set:
                clean_text.write(i)
            clean_text.write("\n")
dataname="test_pegasus_combine_stem"
clean_b(dataname)
