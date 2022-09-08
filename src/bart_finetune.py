import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

class BARTData(torch.utils.data.Dataset):
    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels= labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['label'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)
def token_data(texts,labels,tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = BARTData(encodings, decodings)
    return dataset_tokenized

def fine_tune_bart(dir,train_dir,valid_dir,save_dir,checkdir,freeze_encoder=None):
    checkdir=dir+checkdir
    train_dir=dir+train_dir #dir+'train_finetune.json'
    valid_dir = dir+valid_dir
    save_dir=dir+save_dir
    print('checkdir:'+checkdir)
    print('save_dir:'+save_dir)
    device = 'cuda'
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
    from datasets import load_dataset
    prefix = "summary: "
    dataset = load_dataset('json',data_files={'train': train_dir, 'valid': valid_dir}).shuffle(seed=42)
    train_texts, train_labels = [prefix + each for each in dataset['train']['document']], dataset['train']['summary']
    valid_texts, valid_labels = [prefix + each for each in dataset['valid']['document']], dataset['valid']['summary']
    train_dataset = token_data(train_texts,train_labels,tokenizer)
    valid_dataset = token_data(valid_texts,valid_labels,tokenizer)
    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    batch_size=2
    train_args = Seq2SeqTrainingArguments(
        output_dir=checkdir,
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
        args=train_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,            # evaluation dataset
        tokenizer=tokenizer
    )
    print("training begin:")
    trainer.train()
    trainer.save_model(output_dir=save_dir)
    print("training compelete, output:"+save_dir)
    
    
    
