list=  [['a','aa','aaa'],['b','bbb','ab']]
with open('test.txt','w+') as w:
    for i in list:
        tmpstr=", "
        tmpstr = tmpstr.join(i)
        w.write(tmpstr+"\n")
        
            