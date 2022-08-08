from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json
import torch
import ast
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import nltk
import numpy as np
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize  
from sentence_transformers.cross_encoder import CrossEncoder
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from typing import List