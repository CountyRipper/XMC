from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json
from tqdm import tqdm
from sentence_transformers import CrossEncoder
#from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import nltk
import numpy as np
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize  
from sentence_transformers.cross_encoder import CrossEncoder

from torch.utils.data import DataLoader
from typing import List