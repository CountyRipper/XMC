"""Script for fine-tuning Pegasus
Example usage:
  # use XSum dataset as example, with first 1000 docs as training data
  from datasets import load_dataset
  dataset = load_dataset("xsum")
  train_texts, train_labels = dataset['train']['document'][:1000], dataset['train']['summary'][:1000]
  
  # use Pegasus Large model as base for fine-tuning
  model_name = 'google/pegasus-large'
  train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
  trainer.train()
 
Reference:
  https://huggingface.co/transformers/master/custom_datasets.html

"""
import time #计时器
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
#import nltk
import numpy as np




class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

      
def prepare_data(model_name, 
                 train_texts, train_labels, 
                 val_texts=None, val_labels=None, 
                 test_texts=None, test_labels=None):
  """
  Prepare input data for model fine-tuning
  """
  tokenizer = PegasusTokenizer.from_pretrained(model_name)

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True

  def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized

  train_dataset = tokenize_data(train_texts, train_labels)
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer
'''
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  # Replace -100 in the labels as we can't decode them.
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
  # Rouge expects a newline after each sentence
  decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
  decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
  result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  # Extract a few results
  result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
  # Add mean generated length
  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)
    
  return {k: round(v, 4) for k, v in result.items()}
'''
def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./pegasus-test'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False
  batch_size = 2
  if val_dataset is not None:
    training_args = Seq2SeqTrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=5,           # total number of training epochs
      per_device_train_batch_size=batch_size,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=batch_size,    # batch size for evaluation, can increase if memory allows
      save_steps=30000,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy = "epoch",     # evaluation strategy to adopt during training                 # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
      #compute_metrics=compute_metrics
    )

  else:
    training_args = Seq2SeqTrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=5,           # total number of training epochs
      per_device_train_batch_size=batch_size,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=batch_size,    # batch size for evaluation, can increase if memory allows
      save_steps=30000,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy = "epoch",     # evaluation strategy to adopt during training                 # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
      #compute_metrics=compute_metrics
    )

  return trainer

# if __name__=='__main__':
#   #datadir = "./dataset/"
  
#   # use XSum dataset as example, with first 1000 docs as training data
#   prefix = "summarize: "
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   print(device)
#   from datasets import load_dataset
#   # dataset = load_dataset("xsum")
#   dataset = load_dataset('json',data_files={'train': 'train_finetune.json', 'valid': 'test_finetune.json'}).shuffle(seed=42)

#   train_texts, train_labels = [prefix + each for each in dataset['train']['document']], dataset['train']['summary']
#   valid_texts, valid_labels = [prefix + each for each in dataset['valid']['document']], dataset['valid']['summary']
  
#   # use Pegasus Large model as base for fine-tuning
#   model_name = 'google/pegasus-large'
#   #return train_dataset, val_dataset, test_dataset, tokenizer 可以一起投入
#   train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
#   valid_dataset, _, _, _ = prepare_data(model_name, valid_texts, valid_labels)
#   trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=valid_dataset)
#   #,val_dataset=valid_dataset
#   print("start training")
#   start_time = time.time()
#   trainer.train()
#   trainer.save_model(output_dir='pegasus_test_save')
#   end_time = time.time()
#   print('pegasus_time_cost: ',end_time-start_time,'s')
