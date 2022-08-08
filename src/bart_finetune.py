import yaml
from torch.nn import CrossEntropyLoss, MSELoss
#! pip install datasets transformers rouge-score nltk
import transformers
import datasets
from datasets import load_dataset, load_metric

print(transformers.__version__)
#model_checkpoint = "facebook/bart-base"
model_checkpoint = "facebook/bart-large"

'''
load dataset "kp20k_KPBL_" is what?
'''
#train_dataset = load_dataset('json', data_files='kp20k_BART_train_finetune.src')
#valid_dataset = load_dataset('json', data_files='kp20k_BART_valid_finetune.src')
#test_dataset = load_dataset('json', data_files='kp20k_BART_test_finetune.src')

#raw_datasets = datasets.DatasetDict({"train":train_dataset,"valid":valid_dataset,"test":test_dataset})
raw_datasets = load_dataset('json',data_files={'train': 'kp20k_KPBL_train(1028).json', 'valid': 'kp20k_KPBL_valid(1028).json', 'test': 'kp20k_KPBL_test(1028).json'})
#datasets.DatasetDict.from_json(path_or_paths = 'kp20k_smalltrain_finetune.json') # , features = ['document', 'summary', 'id'])
#raw_datasets = load_dataset(datasets.DatasetDict.from_json(path_or_paths = 'kp20k_smalltrain_finetune.json')) # , features = ['document', 'summary', 'id']))
#from_json
metric = load_metric("rouge")