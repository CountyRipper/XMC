from premethod import *
#from head import*
if __name__ == '__main__':
    # 注意文件路径
    type = ['_labels', '_texts']
    tasks = ['test', 'train']
    dir = './dataset/EUR-Lex/'
    for i in range(len(tasks)):
        label_path = dir+tasks[i]+type[0]
        stem_labels(label_path, label_path+"_stem")
        text_path = dir+tasks[i]+type[1]
        finetune_path = dir+tasks[i]+"_finetune"
        txt_to_json(text_path, label_path+"_stem",
                    finetune_path)  # 注意标签是已经stem过的
    # stem_labels("./dataset/EUR-Lex/test_labels","./dataset/EUR-Lex/test_labels_stem")
    # txt_to_json('./dataset/EUR-Lex/test_texts',"./dataset/EUR-Lex/test_labels_stem","./dataset/EUR-Lex/test_finetune")
