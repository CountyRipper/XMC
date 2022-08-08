from sentence_transformers.cross_encoder import CrossEncoder
import nltk
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

nltk.download('stopwords')

stemmer = SnowballStemmer("english")
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)

model = CrossEncoder('./curr')

same_list = []
src_list = []
# 读取same  test_pegasus.same
with open("test_pegasus_predformat_stem_b.txt", "r+") as same_f:
    for line in same_f:
        same_list.append(line.strip().split(" "))

# 建立result list存结果
result_list = []
scores_list = []
# 读取src 排序 test.src 
with open("test_texts.txt", "r+") as src_f:
    for line in src_f:
        src_list.append(line.strip())

for i in tqdm(range(len(same_list))):
    
    score_list = []
    rank_list = []
    res_list = []
    
    candidates = same_list[i]
    src_text = src_list[i]
    for each_candidate in candidates:
        candidate = each_candidate.replace("_", " ")
        score = model.predict([src_text, candidate])
        score_list.append(score)
        rank_list.append(score)
    rank_list.sort(reverse=True)
    for each_res in rank_list:
        res_list.append(candidates[score_list.index(each_res)])
    result_list.append(res_list)
    scores_list.append([str(x) for x in rank_list])

with open("rank_eurlex4k.txt", "w+") as rank_f:
    for each in result_list:
        rank_f.write(" ".join(each))
        rank_f.write("\n")

with open("rank_eurlex4k.score", "w+") as rank_f:
    for each in scores_list:
        rank_f.write(" ".join(each))
        rank_f.write("\n")
