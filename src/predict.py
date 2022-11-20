from typing import List
from transformers import (BartTokenizer, BartForConditionalGeneration,
                          AutoTokenizer,AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          PegasusForConditionalGeneration, PegasusTokenizer
)
import torch
def batch_pred(model,tokenizer,documents)->List[List]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #if documents==[]:return
    #ARTICLE_TO_SUMMARIZE = document_src
    #加速，集中处理
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)
    #inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
  # Generate Summary
    summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 7).to(device)
    #summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
    pre_result=tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2)
    #pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])  #[2:-2]
    return pre_result
