from collections import defaultdict
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from nltk.corpus import stopwords
from transformers import BertTokenizer, AdamW
from transformers import __version__ as transformers_version
from sklearn.metrics import f1_score
import numpy as np
import os
from tqdm import tqdm
from model import LOTClassModel
import warnings
import datetime
from collections import Counter
import random


warnings.filterwarnings("ignore")

class LOTClassTrainer(object):

    def __init__(self, args):
        self.args = args
        self.max_len = args.max_len
        self.dataset_dir = args.dataset_dir
        # self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.num_cpus = 1
        self.world_size = args.gpus
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.classifier_train_batch_size = args.classifier_train_batch_size
        self.classifier_eval_batch_size = args.classifier_eval_batch_size
        self.accum_steps = args.accum_steps
        eff_batch_size = self.train_batch_size * self.world_size * self.accum_steps
        # assert abs(eff_batch_size - 128) < 10, f"Make sure the effective training batch size is around 128, current: {eff_batch_size}"
        print(f"Effective training batch size: {eff_batch_size}")
        self.pretrained_lm = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        self.read_label_names(args.dataset_dir, args.label_names_file)
        self.num_class = len(self.label_name_dict)
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm, output_attentions=False,
                                                   output_hidden_states=False, num_labels=self.num_class)
        self.lr = args.lr
        self.classifier_lr = args.classifier_lr
        self.classifier_epoch = args.classifier_epoch
        self.classifier_accum_steps = args.classifier_accum_steps
        self.device = torch.device('cuda')
        self.softmax = torch.nn.Softmax(dim=1)
        self.inference_dataloader = None

    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len,
                                                        padding='max_length',
                                                        return_attention_mask=True, truncation=True,
                                                        return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, output=True):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded texts into {loader_file}")
            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks}
            if output == True:
                torch.save(data, loader_file)
        return data

    def create_label_name_data(self, dataset_dir, text_file, label_name_loader_name, output=True):
        loader_file = os.path.join(dataset_dir, label_name_loader_name)
        if os.path.exists(loader_file):
            print(f"Loading texts with label names from {loader_file}")
            label_name_data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            print("Locating label names in the corpus.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(
                delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
            input_ids_with_label_name = torch.cat([result[0] for result in results])
            attention_masks_with_label_name = torch.cat([result[1] for result in results])
            label_name_idx = torch.cat([result[2] for result in results])
            assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
            label_name_data = {"input_ids": input_ids_with_label_name,
                               "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            print(f"Saving texts with label names into {loader_file}")
            if output == True:
                torch.save(label_name_data, loader_file)
        return label_name_data

    # 默认所有label name存在于词典中，而不是unknown word
    def label_name_data_fit_template(self, dataset_dir, label_name_loader_name=None, train_data_with_template=None):
        loader_file = os.path.join(dataset_dir, label_name_loader_name)
        if os.path.exists(loader_file):
            print(f"Loading texts with label names from {loader_file}")
            label_name_data = torch.load(loader_file)
        else:
            input_ids = []
            attention_masks = []
            labels = []     # label indicator

            for i, input_id in enumerate(train_data_with_template["input_ids"]):
                label_indication = np.ones(len(input_id)) * (-1)
                have_class_name = False
                for j, class_name_id in enumerate(self.all_label_name_ids):
                    # ignore [MAKS] token
                    if class_name_id == self.mask_id:
                        continue
                    else:
                        label_index = np.where(input_id == class_name_id)[0]
                        if len(label_index) > 0:
                            label_indication[label_index] = j - 1
                            have_class_name = True
                if have_class_name:
                    input_ids.append(input_id)
                    attention_masks.append(train_data_with_template["attention_masks"][i])
                    # special_token_masks.append(self.train_data_with_pre_post_template["special_token_masks"][i])
                    # doc_labels.append(self.train_data_with_pre_post_template["labels"][i])
                    labels.append(torch.tensor(label_indication).long())

            input_ids = torch.stack(input_ids)
            attention_masks = torch.stack(attention_masks)
            labels = torch.stack(labels)

            label_name_data = {"input_ids": input_ids,
                               "attention_masks": attention_masks,
                               # "special_token_masks": special_token_masks,
                               # "doc_labels": self.train_data_with_pre_post_template["labels"],
                               "labels": labels}
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            print(f"Saving texts with label names into {loader_file}")
            torch.save(label_name_data, loader_file)
        return label_name_data

    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []
        idx = 1  # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1:  # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i + 1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label, add_special_tokens=True,
                                                            max_length=self.max_len,
                                                            padding='max_length', return_attention_mask=True,
                                                            truncation=True, return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower() for word in category_words.strip().split()] for i, category_words in
                                enumerate(label_names)}
        print(f"Label names used for each class are: {self.label_name_dict}")
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size):
        if "labels" in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
        else:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        return dataset_loader

    # filter out stop words and words in multiple categories
    def filter_keywords(self, category_vocab_size=100):
        all_words = defaultdict(list)
        sorted_dicts = {}
        for i, cat_dict in self.category_words_freq.items():
            sorted_dict = {k: v for k, v in
                           sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)
        repeat_words = []
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.category_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.category_vocab[i] = np.array(list(sorted_dict.keys()))
        stopwords_vocab = stopwords.words('english')
        for i, word_list in self.category_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
                if word in self.label_name_dict[i]:
                    continue
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                    delete_idx.append(j)
            self.category_vocab[i] = np.delete(self.category_vocab[i], delete_idx)

    # 扩充label name并保存到label_names_aug.txt，改自category_vocabulary
    def prepare_label_name(self, mode="raw", label_name_count_each_cat=10, top_pred_num=50, category_vocab_size=100, further_pretrain=False):
        if mode == "raw":
            return 0
        elif mode == "aug":
            print(f"Label names augmentation start...")
            label_name_aug_file = os.path.join(self.dataset_dir, self.args.label_names_aug_file)
            if os.path.exists(label_name_aug_file):
                print(f"Label_names_aug.txt is existed. Reading the file...")
                self.read_label_names(self.args.dataset_dir, self.args.label_names_aug_file)
                print(f"Label_names_aug.txt is read.\n")
            else:
                # 读取原始标签，为扩充标签做准备
                label_name_data = self.create_label_name_data(dataset_dir=self.args.dataset_dir, text_file=self.args.train_file, label_name_loader_name="label_name_data.pt")
                print("Contructing category vocabulary.")

                model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                       output_attentions=False,
                                                       output_hidden_states=False,
                                                       num_labels=self.num_class)
                if further_pretrain:
                    model = self.further_pretrain(model=model, epoch=1, save_model=True)

                model.to(self.device)
                model.eval()
                label_name_dataset_loader = self.make_dataloader(0, label_name_data, self.eval_batch_size)
                category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}

                for batch in tqdm(label_name_dataset_loader):
                    with torch.no_grad():
                        input_ids = batch[0].to(self.device)
                        input_mask = batch[1].to(self.device)
                        label_pos = batch[2].to(self.device)
                        match_idx = label_pos >= 0
                        predictions = model(input_ids,
                                            pred_mode="mlm_inference",
                                            token_type_ids=None,
                                            attention_mask=input_mask)
                        _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)
                        label_idx = label_pos[match_idx]
                        for i, word_list in enumerate(sorted_res):
                            for j, word_id in enumerate(word_list):
                                category_words_freq[label_idx[i].item()][word_id.item()] += 1
                    torch.cuda.empty_cache()

                self.category_words_freq = category_words_freq
                self.filter_keywords(category_vocab_size)

                # print augmented label names to label_names_aug.txt
                for i, category_vocab in self.category_vocab.items():
                    category_vocab_list = list(category_vocab)
                    label_names = self.label_name_dict[i]
                    rest_count = label_name_count_each_cat - len(label_names)
                    while rest_count > 0:
                        top_word = self.inv_vocab[category_vocab_list.pop(0)]
                        if top_word not in label_names:
                            label_names.append(top_word)
                            rest_count -= 1
                    f_out = open(label_name_aug_file, 'a+')
                    for word in label_names:
                        f_out.write(f"{word} ")
                    f_out.write("\n")
                    f_out.close()

                # 重新读取augmented label names
                self.read_label_names(self.args.dataset_dir, self.args.label_names_aug_file)

    def further_pretrain(self, model, rank=0, epoch=1, save_model=True):
        for_mlm_without_template = {}
        self.train_data = self.create_dataset(self.args.dataset_dir, self.args.train_file, self.args.train_label_file, "train.pt")
        for_mlm_without_template['input_ids'], for_mlm_without_template['mlm_labels'] = \
            self.mask_tokens_for_mlm(self.train_data["input_ids"])
        train_set_for_mlm = TensorDataset(for_mlm_without_template['input_ids'],
                                          self.train_data["attention_masks"],
                                          for_mlm_without_template['mlm_labels'])
        train_set_for_mlm_dataloader = DataLoader(train_set_for_mlm, batch_size=24, shuffle=False)

        optim = AdamW(self.model.parameters(), lr=1e-5)

        model.to(rank)
        model.train()

        torch.cuda.empty_cache()
        for i in range(epoch):
            for batch in tqdm(train_set_for_mlm_dataloader):

                optim.zero_grad()
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                masked_token_labels = batch[2].to(rank)
                loss = model(input_ids, pred_mode="mlm", attention_mask=input_mask, mlm_labels=masked_token_labels)
                loss.backward()

                # if (count + 1) % self.accum_steps == 0:
                optim.step()
                optim.zero_grad()
                torch.cuda.empty_cache()

        if save_model:
            file_name = "fp_BERT.pt"
            loader_file = os.path.join(self.dataset_dir, file_name)
            torch.save(self.model.state_dict(), loader_file)

        return model


    def inference2(self, model, rank, mode="testset_template", label_embed_mode="seperate"):

        print(f"label name embedding calculation start ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")
        # calculate the embeddings of label names (using training samples)

        label_name_included_text_set = TensorDataset(self.label_name_data_with_template["input_ids"], self.label_name_data_with_template["attention_masks"], self.label_name_data_with_template["labels"])
        label_name_included_text_dataloader = DataLoader(label_name_included_text_set, sampler=SequentialSampler(label_name_included_text_set), batch_size=self.eval_batch_size)

        model.to(rank)
        model.eval()

        if label_embed_mode == "mix":
            # initialize the list to store label embeddings
            label_emb_matrix = []
            for i in range(len(self.label_name_dict)):
                exec('label_emb_store_{} = []'.format(i))

        elif label_embed_mode == "seperate":
            # initialize the list to store label embeddings
            label_emb_matrix = []
            for i in range(len(self.label2class)):
                exec('label_emb_store_{} = []'.format(i))

        for batch in label_name_included_text_dataloader:
            with torch.no_grad():
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                last_hidden_state = model(input_ids, pred_mode="inference", token_type_ids=None,attention_mask=input_mask)
                label_position = batch[2].to(rank)

                match_idx = label_position >= 0
                label_idx = label_position[match_idx]
                last_hidden_state_of_label = last_hidden_state[match_idx]

                for i, l_idx in enumerate(label_idx):
                    if label_embed_mode == "mix":
                        # average embeddings if more than one word represent the category
                        label_id = self.all_label_name_ids[l_idx.item() + 1]
                        label_name = self.inv_vocab[label_id]
                        category_idx = self.label2class[label_name]
                    elif label_embed_mode == "seperate":
                        category_idx = l_idx
                    exec('label_emb_store_{}.append(last_hidden_state_of_label[{}])'.format(int(category_idx), i))

        if label_embed_mode == "mix":
            for i in range(len(self.label_name_dict)):
                exec('label_emb_matrix.append(torch.mean(torch.stack(label_emb_store_{}), dim=0))'.format(i))

        elif label_embed_mode == "seperate":
            for i in range(len(self.label2class)):
                exec('label_emb_matrix.append(torch.mean(torch.stack(label_emb_store_{}), dim=0))'.format(i))

        label_emb_matrix = torch.stack(label_emb_matrix)

        print(f"label name embedding calculation finished ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

        if mode == "testset_template":
            test_with_template = TensorDataset(self.test_data_with_template["input_ids"], self.test_data_with_template["attention_masks"], self.test_data_with_template["labels"])
            inference_dataloader = DataLoader(test_with_template, sampler=SequentialSampler(test_with_template), batch_size=self.eval_batch_size)
            truth_labels = []

        elif mode == "trainset_template":
            train_with_template = TensorDataset(self.train_data_with_template["input_ids"], self.train_data_with_template["attention_masks"])
            inference_dataloader = DataLoader(train_with_template, sampler=SequentialSampler(train_with_template), batch_size=self.eval_batch_size)

        elif mode == "confidence_factor":
            # 固定种子，从而固定每次计算CF的数据
            if self.inference_dataloader is None:
                id_for_cf = torch.floor(torch.rand(10000) * self.train_data_with_template['input_ids'].size()[0]).long()
                input_ids_cf = torch.index_select(self.train_data_with_template["input_ids"], dim=0, index=id_for_cf)
                attention_masks_cf = torch.index_select(self.train_data_with_template["attention_masks"], dim=0, index=id_for_cf)
                train_with_template = TensorDataset(input_ids_cf, attention_masks_cf)
                self.inference_dataloader = DataLoader(train_with_template, sampler=SequentialSampler(train_with_template), batch_size=self.eval_batch_size)
            inference_dataloader = self.inference_dataloader

        doc_pred = []
        for batch in inference_dataloader:
            with torch.no_grad():
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                mask_idx = input_ids == 103
                last_hidden_state = model(input_ids, pred_mode="inference", token_type_ids=None, attention_mask=input_mask)

                doc_embs = last_hidden_state[mask_idx]
                cos_matrix = self.Cosine(doc_embs, label_emb_matrix)
                doc_pred.append(cos_matrix)

                if mode == "testset" or mode == "testset_template":
                    truth_labels.append(batch[2])

        doc_pred = torch.cat(doc_pred, dim=0)
        # choose the label name having max similarity score if one category has multiple label names
        if label_embed_mode == "seperate":
            doc_pred_numpy = doc_pred.cpu().numpy()
            new_doc_pred = np.zeros((doc_pred_numpy.shape[0], len(self.label_name_dict)))
            for key in self.label_name_dict.keys():
                if len(self.label_name_dict[key]) == 1:
                    ln = self.label_name_dict[key][0]
                    label_index = self.all_label_names.index(ln) - 1   # The first element is [MASK]
                    new_doc_pred[:,key] = doc_pred_numpy[:, label_index]
                elif len(self.label_name_dict[key]) > 1:
                    ln_indexes = [self.all_label_names.index(ln) - 1 for ln in self.label_name_dict[key]]
                    new_doc_pred[:, key] = doc_pred_numpy[:, ln_indexes].max(axis=1)
            doc_pred = torch.tensor(new_doc_pred)

        doc_pred_label = torch.argmax(doc_pred, dim=1)

        if mode == "testset_template":
            truth_labels = torch.cat(truth_labels, dim=0)
            macro_f1 = f1_score(truth_labels, doc_pred_label.to("cpu"), average='macro')
            micro_f1 = f1_score(truth_labels, doc_pred_label.to("cpu"), average='micro')
            return macro_f1, micro_f1, doc_pred, doc_pred_label

        elif mode == "trainset_template" or mode == "confidence_factor":
            if self.num_class == 2:
                mu = torch.mean(doc_pred, dim=1).unsqueeze(1).repeat(1, doc_pred.size()[1])
                doc_pred_normalized = doc_pred - mu
                soft_result_pt = self.softmax(doc_pred_normalized)
                confidence = torch.max(soft_result_pt, dim=1)
                cf = float(torch.mean(confidence[0]))
                return cf, doc_pred, doc_pred_label
            else:
                # Z-score normalization
                mu = torch.mean(doc_pred, dim=1).unsqueeze(1).repeat(1, doc_pred.size()[1])
                sigma = torch.std(doc_pred, dim=1).unsqueeze(1).repeat(1, doc_pred.size()[1])
                doc_pred_normalized = (doc_pred - mu) / sigma
                soft_result_pt = self.softmax(doc_pred_normalized)
                confidence = torch.max(soft_result_pt, dim=1)
                cf = float(torch.mean(confidence[0]))
                return cf, doc_pred_normalized, doc_pred_label

    def inference3(self, model, rank, label_embed_mode="seperate"):

        print(f"label name embedding calculation start ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

        label_name_included_text_set = TensorDataset(self.label_name_data_with_template["input_ids"], self.label_name_data_with_template["attention_masks"], self.label_name_data_with_template["labels"])
        label_name_included_text_dataloader = DataLoader(label_name_included_text_set, sampler=SequentialSampler(label_name_included_text_set), batch_size=self.eval_batch_size)

        model.to(rank)
        model.eval()

        if label_embed_mode == "mix":
            # initialize the list to store label embeddings
            label_emb_matrix = []
            for i in range(len(self.label_name_dict)):
                exec('label_emb_store_{} = []'.format(i))

        elif label_embed_mode == "seperate":
            # initialize the list to store label embeddings
            label_emb_matrix = []
            for i in range(len(self.label2class)):
                exec('label_emb_store_{} = []'.format(i))

        for batch in label_name_included_text_dataloader:
            with torch.no_grad():
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                last_hidden_state = model(input_ids, pred_mode="inference", token_type_ids=None,attention_mask=input_mask)
                label_position = batch[2].to(rank)

                match_idx = label_position >= 0
                label_idx = label_position[match_idx]
                last_hidden_state_of_label = last_hidden_state[match_idx]

                for i, l_idx in enumerate(label_idx):
                    if label_embed_mode == "mix":
                        # average embeddings if more than one word represent the category
                        label_id = self.all_label_name_ids[l_idx.item() + 1]
                        label_name = self.inv_vocab[label_id]
                        category_idx = self.label2class[label_name]
                    elif label_embed_mode == "seperate":
                        category_idx = l_idx
                    exec('label_emb_store_{}.append(last_hidden_state_of_label[{}])'.format(int(category_idx), i))

        if label_embed_mode == "mix":
            for i in range(len(self.label_name_dict)):
                exec('label_emb_matrix.append(torch.mean(torch.stack(label_emb_store_{}), dim=0))'.format(i))

        elif label_embed_mode == "seperate":
            for i in range(len(self.label2class)):
                exec('label_emb_matrix.append(torch.mean(torch.stack(label_emb_store_{}), dim=0))'.format(i))

        label_emb_matrix = torch.stack(label_emb_matrix)

        print(f"label name embedding calculation finished ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

        # 计算在测试集的标签
        test_with_template = TensorDataset(self.test_data_with_template["input_ids"], self.test_data_with_template["attention_masks"], self.test_data_with_template["labels"])
        inference_dataloader_test_with_template = DataLoader(test_with_template, sampler=SequentialSampler(test_with_template), batch_size=self.eval_batch_size)
        truth_labels = []

        doc_pred_test = []
        for batch in inference_dataloader_test_with_template:
            with torch.no_grad():
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                mask_idx = input_ids == 103
                last_hidden_state = model(input_ids, pred_mode="inference", token_type_ids=None, attention_mask=input_mask)
                doc_embs = last_hidden_state[mask_idx]
                cos_matrix = self.Cosine(doc_embs, label_emb_matrix)
                doc_pred_test.append(cos_matrix)
                truth_labels.append(batch[2])
        doc_pred_test = torch.cat(doc_pred_test, dim=0)

        if label_embed_mode == "seperate":
            doc_pred_numpy = doc_pred_test.cpu().numpy()
            new_doc_pred = np.zeros((doc_pred_numpy.shape[0], len(self.label_name_dict)))
            for key in self.label_name_dict.keys():
                if len(self.label_name_dict[key]) == 1:
                    ln = self.label_name_dict[key][0]
                    label_index = self.all_label_names.index(ln) - 1   # The first element is [MASK]
                    new_doc_pred[:,key] = doc_pred_numpy[:, label_index]
                elif len(self.label_name_dict[key]) > 1:
                    ln_indexes = [self.all_label_names.index(ln) - 1 for ln in self.label_name_dict[key]]
                    new_doc_pred[:, key] = doc_pred_numpy[:, ln_indexes].max(axis=1)
            doc_pred_test = torch.tensor(new_doc_pred)
        doc_pred_test_label = torch.argmax(doc_pred_test, dim=1)

        truth_labels = torch.cat(truth_labels, dim=0)
        macro_f1_test = f1_score(truth_labels, doc_pred_test_label.to("cpu"), average='macro')
        micro_f1_test = f1_score(truth_labels, doc_pred_test_label.to("cpu"), average='micro')

        torch.cuda.empty_cache()

        # 计算训练集上的CF
        # 固定种子，从而固定每次计算CF的数据
        if self.inference_dataloader is None:
            id_for_cf = torch.floor(torch.rand(10000) * self.train_data_with_template['input_ids'].size()[0]).long()
            input_ids_cf = torch.index_select(self.train_data_with_template["input_ids"], dim=0, index=id_for_cf)
            attention_masks_cf = torch.index_select(self.train_data_with_template["attention_masks"], dim=0,
                                                    index=id_for_cf)
            train_with_template = TensorDataset(input_ids_cf, attention_masks_cf)
            self.inference_dataloader = DataLoader(train_with_template, sampler=SequentialSampler(train_with_template),
                                                   batch_size=self.eval_batch_size)
        inference_dataloader_train_with_template = self.inference_dataloader

        # train_with_template = TensorDataset(self.train_data_with_pre_post_template["input_ids"], self.train_data_with_pre_post_template["attention_masks"])
        # inference_dataloader_train_with_template = DataLoader(train_with_template, sampler=SequentialSampler(train_with_template), batch_size=self.eval_batch_size)

        doc_pred_train = []
        for batch in inference_dataloader_train_with_template:
            with torch.no_grad():
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                mask_idx = input_ids == 103
                last_hidden_state = model(input_ids, pred_mode="inference", token_type_ids=None, attention_mask=input_mask)
                doc_embs = last_hidden_state[mask_idx]
                cos_matrix = self.Cosine(doc_embs, label_emb_matrix)
                doc_pred_train.append(cos_matrix)
        doc_pred_train = torch.cat(doc_pred_train, dim=0)

        if label_embed_mode == "seperate":
            doc_pred_numpy = doc_pred_train.cpu().numpy()
            new_doc_pred = np.zeros((doc_pred_numpy.shape[0], len(self.label_name_dict)))
            for key in self.label_name_dict.keys():
                if len(self.label_name_dict[key]) == 1:
                    ln = self.label_name_dict[key][0]
                    label_index = self.all_label_names.index(ln) - 1   # The first element is [MASK]
                    new_doc_pred[:,key] = doc_pred_numpy[:, label_index]
                elif len(self.label_name_dict[key]) > 1:
                    ln_indexes = [self.all_label_names.index(ln) - 1 for ln in self.label_name_dict[key]]
                    new_doc_pred[:, key] = doc_pred_numpy[:, ln_indexes].max(axis=1)
            doc_pred_train = torch.tensor(new_doc_pred)

        if self.num_class == 2:
            mu = torch.mean(doc_pred_train, dim=1).unsqueeze(1).repeat(1, doc_pred_train.size()[1])
            doc_pred_normalized = doc_pred_train - mu
            soft_result_pt = self.softmax(doc_pred_normalized)
            confidence = torch.max(soft_result_pt, dim=1)
            cf = float(torch.mean(confidence[0]))
        else:
            mu = torch.mean(doc_pred_train, dim=1).unsqueeze(1).repeat(1, doc_pred_train.size()[1])
            sigma = torch.std(doc_pred_train, dim=1).unsqueeze(1).repeat(1, doc_pred_train.size()[1])
            doc_pred_normalized = (doc_pred_train - mu) / sigma
            soft_result_pt = self.softmax(doc_pred_normalized)
            confidence = torch.max(soft_result_pt, dim=1)
            cf = float(torch.mean(confidence[0]))

        return cf, macro_f1_test, micro_f1_test

    def Cosine(self, doc_embs, label_embs):
        """
        :param doc_embs: m x k tensor
        :param label_embs: n x k tensor
        :return: m x n tensor
        """
        xx = torch.sum(doc_embs ** 2, dim=1) ** 0.5
        doc_embs = doc_embs / xx.unsqueeze(dim=1)
        yy = torch.sum(label_embs ** 2, dim=1) ** 0.5
        label_embs = label_embs / yy.unsqueeze(dim=1)
        return torch.mm(doc_embs, torch.transpose(label_embs, 0, 1))

    def text_fit_template(self, dataset_dir, text_file, label_file, loader_name, pre_temp, post_temp):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading template encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"template fit start ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            pre_template = pre_temp.replace('*mask*', self.tokenizer.mask_token)
            post_template = post_temp.replace('*mask*', self.tokenizer.mask_token)
            pre_template_encode = self.encode([pre_template])
            post_template_encode = self.encode([post_template])

            # get ids of pre template ([cls] included)
            pre_template_ids_length = int(pre_template_encode[1].sum(dim=1)[0]) - 1
            pre_template_ids = pre_template_encode[0][0][0:pre_template_ids_length]

            # get ids of post template ([sep] included)
            post_template_ids_length = int(post_template_encode[1].sum(dim=1)[0]) - 1
            post_template_ids = post_template_encode[0][0][1:post_template_ids_length + 1]

            docs = [doc.strip() for doc in corpus.readlines()]

            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
            main_texts = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids_list = []
            attention_mask_list = []
            special_token_mask_list = []

            for main_text in main_texts:
                main_text_length = main_text[1].sum(dim=1) - 2
                # truncation if length exceed 512
                total_length = pre_template_ids_length + post_template_ids_length + main_text_length
                for i, each in enumerate(main_text[0]):
                    if total_length[i] > 512:
                        main_text_length[i] = 512 - pre_template_ids_length - post_template_ids_length
                        total_length[i] = 512

                    main_text_ids = main_text[0][i][1:main_text_length[i] + 1]
                    template_result = torch.tensor(list(pre_template_ids) + list(main_text_ids) + list(post_template_ids))
                    template_result_i = torch.zeros(512)
                    template_result_i[0:template_result.size()[0]] = template_result
                    input_ids_list.append(template_result_i)

                    attention_mask = torch.ones(total_length[i])
                    attention_mask_i = torch.zeros(512)
                    attention_mask_i[0:attention_mask.size()[0]] = attention_mask
                    attention_mask_list.append(attention_mask_i)

                    special_token_mask_i = torch.ones(512)
                    special_token_mask_i[pre_template_ids_length:pre_template_ids_length + main_text_length[i]] = 0
                    special_token_mask_list.append(special_token_mask_i)

            input_ids = torch.stack(input_ids_list).long()
            attention_masks = torch.stack(attention_mask_list).long()
            special_token_masks = torch.stack(special_token_mask_list).long()

            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "special_token_masks": special_token_masks, "labels": labels}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "special_token_masks": special_token_masks}

            print(
                f"Saving encoded texts into {loader_file} ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")
            torch.save(data, loader_file)
        return data

    def write_results_multi_template(self, out_file="out.txt", template_count=None, mode=None, label_mode="raw"):

        results_list = []
        macros = []
        micros = []
        CFs = []

        for template_id in range(template_count):
            print(f"Template {template_id} evaluation start ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

            exec(f"self.pre_template = self.args.pre_template_{template_id}")
            exec(f"self.post_template = self.args.post_template_{template_id}")

            if label_mode == "aug":
                self.read_label_names(self.args.dataset_dir, self.args.label_names_aug_file)

            self.train_data_with_template = self.text_fit_template(self.dataset_dir,
                                                                   self.args.train_file,
                                                                   self.args.train_label_file,
                                                                   f'train_with_template_{template_id}.pt',
                                                                   pre_temp=self.pre_template,
                                                                   post_temp=self.post_template)
            self.test_data_with_template = self.text_fit_template(self.dataset_dir,
                                                                  self.args.test_file,
                                                                  self.args.test_label_file,
                                                                  f'test_with_template_{template_id}.pt',
                                                                  pre_temp=self.pre_template,
                                                                  post_temp=self.post_template)
            self.label_name_data_with_template = self.label_name_data_fit_template(self.dataset_dir,
                                                                                   label_name_loader_name=f'label_name_data_with_pre_post_template_{template_id}.pt',
                                                                                   train_data_with_template=self.train_data_with_template)

            model_name = f"final_model_{template_id}.pt"
            loader_file = os.path.join(self.dataset_dir, model_name)
            print(f"\nLoading final model from {loader_file}")
            self.model.load_state_dict(torch.load(loader_file))
            self.model.to(0)

            torch.cuda.empty_cache()
            if mode == "testset_template":
                macro_f1, micro_f1, doc_pred, doc_pred_label = self.inference2(self.model, rank=0, mode="testset_template")
                macros.append(macro_f1)
                micros.append(micro_f1)
            elif mode == "trainset_template":
                confidence_score, doc_pred, doc_pred_label = self.inference2(self.model, rank=0, mode="trainset_template")
                CFs.append(confidence_score)

            torch.cuda.empty_cache()
            results_list.append(doc_pred)

        doc_pred_ensemble = sum(results_list) / len(results_list)
        doc_pred_label_ensemble = torch.argmax(doc_pred_ensemble, dim=1)

        if mode == "testset_template":
            truth_labels = self.test_data_with_template["labels"]
        elif mode == "trainset_template":
            truth_labels = self.train_data_with_template["labels"]

        macro_f1 = f1_score(truth_labels, doc_pred_label_ensemble.to("cpu"), average='macro')
        micro_f1 = f1_score(truth_labels, doc_pred_label_ensemble.to("cpu"), average='micro')

        if mode == "testset_template":
            out_file = os.path.join(self.dataset_dir, out_file)
            print(f"The macro f1 is {macro_f1}, micro f1 is {micro_f1}.")
            print(f"Writing prediction results to {out_file}")
            f_out = open(out_file, 'a+')
            f_out.write(f"\n{datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Epoch: {self.args.epoch}  Learning rate: {self.lr}  Train batch: {self.train_batch_size}  "
                        f"Accum step: {self.accum_steps}  Train data: {self.train_data_with_template['input_ids'].size()[0]}  "
                        f"cl loss weight: {self.args.cl_loss_weight}  mlm loss weight: {self.args.mlm_loss_weight} "
                        f"Save prompt model checkpoint every {self.args.save_prompt_step} steps "
                        f"label name mode: {self.args.label_name_mode} label name count: {self.args.label_name_aug_count}\n"
                        f"{self.args.save_prompt_step}\n{self.accum_steps}\n{self.args.prompt_early_stop}\n"
                        f"{self.lr}\n{self.args.label_name_aug_count}\n{self.args.cl_loss_weight}\n{self.args.mlm_loss_weight}\n")
            for ma, mi in zip(macros, micros):
                f_out.write(f"Prompt classifier: Macro-F1: {ma}, Micro-F1: {mi}\n")
            f_out.write(f"Ensembled prompt classifier:  Macro-F1: {str(float(macro_f1))}, Micro-F1: {str(float(micro_f1))}\n\n")
            f_out.close()

        if mode == "testset_template":
            return macro_f1, micro_f1, doc_pred_ensemble, doc_pred_label_ensemble
        elif mode == "trainset_template":
            return macro_f1, micro_f1, doc_pred_ensemble, doc_pred_label_ensemble, CFs

    def save_model(self, file_name="final_model.pt"):
        loader_file = os.path.join(self.dataset_dir, file_name)
        torch.save(self.model.state_dict(), loader_file)

    # multi-template
    def cl_train_multi_template(self, rank, lr, epoch, template_id):
        print(f"Template {template_id} CL training start ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

        exec(f"self.pre_template = self.args.pre_template_{template_id}")
        exec(f"self.post_template = self.args.post_template_{template_id}")

        self.train_data_with_template = self.text_fit_template(self.dataset_dir, self.args.train_file, self.args.train_label_file, f'train_with_template_{template_id}.pt', pre_temp=self.pre_template, post_temp=self.post_template)
        self.test_data_with_template = self.text_fit_template(self.dataset_dir, self.args.test_file, self.args.test_label_file, f'test_with_template_{template_id}.pt', pre_temp=self.pre_template, post_temp=self.post_template)
        self.label_name_data_with_template = self.label_name_data_fit_template(self.dataset_dir, label_name_loader_name=f'label_name_data_with_pre_post_template_{template_id}.pt', train_data_with_template=self.train_data_with_template)

        # initiate the model
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm, output_attentions=False, output_hidden_states=False, num_labels=self.num_class)

        output_file = os.path.join(self.dataset_dir, self.args.out_file)
        f_out = open(output_file, 'a+')
        f_out.write(f"prompt train start ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})\n"
                    f"Epoch: {self.args.epoch}  Learning rate: {self.lr}  Train batch: {self.train_batch_size}  "
                    f"Accum step: {self.accum_steps} Prompt early stop: {self.args.prompt_early_stop} "
                    f"Train data: {self.train_data_with_template['input_ids'].size()[0]}  "
                    f"cl loss weight: {self.args.cl_loss_weight}  mlm loss weight: {self.args.mlm_loss_weight} "
                    f"pre_template: {self.pre_template}  post_template: {self.post_template} "
                    f"Save prompt model checkpoint every {self.args.save_prompt_step} steps\n")
        f_out.close()

        for_mlm_with_template = {}
        for_mlm_with_template['input_ids'], for_mlm_with_template['mlm_labels'] = \
            self.mask_tokens_for_mlm(self.train_data_with_template["input_ids"], self.train_data_with_template["special_token_masks"])
        train_set_for_mlm = TensorDataset(for_mlm_with_template['input_ids'], self.train_data_with_template["attention_masks"],
                                          for_mlm_with_template['mlm_labels'], self.train_data_with_template["special_token_masks"])
        train_set_for_mlm_dataloader = DataLoader(train_set_for_mlm, batch_size=self.train_batch_size, shuffle=True)

        optim = AdamW(self.model.parameters(), lr=lr)

        self.model.to(rank)
        self.model.train()

        torch.cuda.empty_cache()
        for i in range(epoch):
            train_loss = 0
            count = 1
            max_train_confidence = 0
            early_stop_count = 0
            for batch in tqdm(train_set_for_mlm_dataloader):

                optim.zero_grad()
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                masked_token_labels = batch[2].to(rank)
                template_tokens = batch[3].to(rank)
                cl_loss, mlm_loss = self.model(input_ids, pred_mode="cl+mlm2", attention_mask=input_mask, mlm_labels=masked_token_labels, template_tokens=template_tokens)
                torch.cuda.empty_cache()

                loss = (self.args.cl_loss_weight * cl_loss + self.args.mlm_loss_weight * mlm_loss) / self.accum_steps
                train_loss = train_loss + loss.item()
                loss.backward()

                if (count + 1) % self.accum_steps == 0:
                    optim.step()
                    optim.zero_grad()

                count = count + 1

                # save the model every 250 batches
                if count % self.args.save_prompt_step == 0:

                    train_confidence, test_macro, test_micro = self.inference3(model=self.model, rank=0)

                    output_file = os.path.join(self.dataset_dir, self.args.out_file)
                    f_out = open(output_file, 'a+')
                    idx = count / self.args.save_prompt_step
                    f_out.write(f'{int(idx)}, confidence score:{train_confidence}, '
                                f'Macro-F1-test:{str(float(test_macro))}, Micro-F1-test:{str(float(test_micro))}\n')
                    f_out.close()

                    if train_confidence > max_train_confidence:
                        early_stop_count = 0
                        max_train_confidence = train_confidence
                        exec(f"self.save_model(file_name='final_model_{template_id}.pt')")
                    else:
                        early_stop_count += 1

                if early_stop_count > self.args.prompt_early_stop:
                    break

            print(f"Epoch {i} loss: {train_loss}")

        print(f"CL training finished ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

        print(f"LM is saved to 'final_model_{template_id}.pt' ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")
        torch.cuda.empty_cache()

    def mask_tokens_for_mlm(self, input_ids, special_tokens_mask=None):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = input_ids.clone()
        input_ids_copy = input_ids.clone() # prevent change the orignal training ids
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        if special_tokens_mask == None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val
                                   in
                                   labels.tolist()]
        else:
            special_tokens_mask = special_tokens_mask.int().tolist()

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')[:3]] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids_copy[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids_copy[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids_copy, labels

    # def check_point_eval(self):
    #     check_point_path = os.path.join(self.dataset_dir, 'checkpoints')
    #     file_list = os.listdir(check_point_path)
    #     file_list.sort()
    #     output_file = os.path.join(check_point_path, 'checkpoint_result.txt')
    #
    #     f_out = open(output_file, 'a+')
    #     f_out.write(f"{datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
    #                 f"Epoch: {self.args.epoch}  Learning rate: {self.args.lr}  Train batch: {self.train_batch_size}  "
    #                 f"Accum step: {self.accum_steps}  Train data: {self.train_data['input_ids'].size()[0]}"
    #                 f"cl loss weight: {self.args.cl_loss_weight}  mlm loss weight: {self.args.mlm_loss_weight} "
    #                 f"Save prompt model checkpoint every {self.args.save_prompt_step} steps\n")
    #     f_out.close()
    #
    #     for model_name in file_list:
    #         if model_name != 'checkpoint_result.txt':
    #             check_point = os.path.join(check_point_path, model_name)
    #             model = torch.load(check_point)
    #             macro_f1, micro_f1, _, _ = self.inference2(model, rank=0, mode="testset_template")
    #             f_out = open(output_file, 'a+')
    #             f_out.write(model_name + f'   Macro-F1: ' + str(float(macro_f1)) + '  Micro-F1: ' + str(float(micro_f1)) + '\n')
    #             f_out.close()
    #
    #     f_out = open(output_file, 'a+')
    #     f_out.write(f'\n\n')
    #     f_out.close()

    def train_classifier_selftrain_multi_template(self):

        out_file = os.path.join(self.dataset_dir, self.args.out_file)
        f_out = open(out_file, 'a+')
        f_out.write(f"Classifier selftraining start {datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Classifier epoch: {self.classifier_epoch}  Selftraining learning rate: {self.classifier_lr} "
                    f"Prompt threshold: {self.args.prompt_thresh}  Classifier threshold: {self.args.classifier_thresh}"
                    f"Selftrain batch: {self.args.classifier_train_batch_size} "
                    f"Classifier accum step: {self.args.classifier_accum_steps} "
                    f"trainset per: {self.args.train_per}\n"
                    f"Template classifier\n{self.args.prompt_thresh}\n{self.args.classifier_thresh}\n{self.classifier_lr}\n"
                    f"{self.classifier_epoch}\n{self.args.classifier_train_batch_size}\n"
                    f"{self.args.classifier_accum_steps}\n{self.args.train_per}\n\n")
        f_out.close()

        # self training start
        for self_iter in range(self.args.self_training_max_time):
            print(f"Self training iteration {self_iter} start...")

            f_out = open(out_file, 'a+')
            f_out.write(f"\nSelf training {self_iter} start...")
            f_out.close()

            # confidence set comes from promptclassifier at the initial stage
            if self_iter == 0:
                _, _, doc_pred_processed, doc_pred_label, confidence_factors = self.write_results_multi_template(
                    out_file=self.args.out_file, template_count=self.args.template_count, mode="trainset_template")

                # 根据confidence factor选择模板
                target_template_id = confidence_factors.index(max(confidence_factors))
                print(f"Template{target_template_id} is used for classifier finetuning.\n Confidence factors are {str(confidence_factors)}")

                f_out = open(out_file, 'a+')
                f_out.write(f"\nTemplate{target_template_id} is used for classifier finetuning.\n Confidence factors are {str(confidence_factors)}\n")
                f_out.close()

                loader_file = os.path.join(self.dataset_dir, f"final_model_{target_template_id}.pt")
                self.model.load_state_dict(torch.load(loader_file))

                self.train_data_with_template = self.text_fit_template(self.dataset_dir,
                                                                       self.args.train_file,
                                                                       self.args.train_label_file,
                                                                       f'train_with_template_{target_template_id}.pt',
                                                                       pre_temp=self.pre_template,
                                                                       post_temp=self.post_template)
                self.test_data_with_template = self.text_fit_template(self.dataset_dir,
                                                                      self.args.test_file,
                                                                      self.args.test_label_file,
                                                                      f'test_with_template_{target_template_id}.pt',
                                                                      pre_temp=self.pre_template,
                                                                      post_temp=self.post_template)
                self.label_name_data_with_template = self.label_name_data_fit_template(self.dataset_dir,
                                                                                       label_name_loader_name=f'label_name_data_with_pre_post_template_{target_template_id}.pt',
                                                                                       train_data_with_template=self.train_data_with_template)

            # confidence set comes from BERTclassifier at the initial stage
            else:
                _, _, doc_pred, doc_pred_label = self.eval_classifier(mode="trainset_template")
                doc_pred_processed = torch.from_numpy(doc_pred).to(self.device)
                doc_pred_label = torch.from_numpy(doc_pred_label).to(self.device)

            softmaxed_pred_pt = self.softmax(doc_pred_processed)
            confidence = torch.max(softmaxed_pred_pt, dim=1)
            count = Counter(confidence[1].tolist())

            confidence_np = confidence[0].cpu().numpy()

            # find confidence predictions and built train set dataloader and val set dataloader
            if self_iter == 0:
                confidence_ids = torch.tensor(np.where(confidence_np > self.args.prompt_thresh)[0])
            else:
                confidence_ids = torch.tensor(np.where(confidence_np > self.args.classifier_thresh)[0])

            # save evaluation result of confidence set
            pseudo_label_confidence_set = torch.index_select(doc_pred_label.to("cpu"), dim=0, index=confidence_ids).numpy()
            true_label_confidence_set = torch.index_select(self.train_data_with_template["labels"], dim=0, index=confidence_ids).numpy()
            confidence_set_macro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='macro')
            confidence_set_micro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='micro')

            # explore the different thresh
            candidate_threshs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999]
            self.test_thresh(confidence_np=confidence_np, doc_pred_label=doc_pred_label, candidate_threshs=candidate_threshs)

            train_dataloader, val_dataloader, each_cat_count = self.prepare_selftrain_data(self.train_data_with_template,
                                                                                           confidence_ids, doc_pred_label, softmaxed_pred_pt, train_per=self.args.train_per, min_sample_per_class=10, pseudo_label_mode=self.args.pseudo_label_mode)

            f_out = open(out_file, 'a+')
            f_out.write(f"{str(count)}\nConfidence size: {confidence_ids.size()[0]}, Each cat count: {each_cat_count}, Macro-F1: "
                        f"{str(float(confidence_set_macro_f1))}, Micro-F1: {str(float(confidence_set_micro_f1))}\n"
                        f"{confidence_ids.size()[0]}/{each_cat_count},{str(float(confidence_set_macro_f1))},{str(float(confidence_set_micro_f1))}\n")
            f_out.close()

            optim = AdamW(self.model.parameters(), lr=self.classifier_lr)
            rank = self.device

            self.model.to(0)
            self.model.train()

            torch.cuda.empty_cache()

            min_val_loss = 1e10
            early_stop_count = 0

            for epoch in range(self.classifier_epoch):
                train_loss = 0
                count = 0
                for batch in train_dataloader:

                    optim.zero_grad()
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    labels = batch[2].to(rank)
                    _, classifier_loss = self.model(input_ids, pred_mode="classifier", attention_mask=input_mask, labels=labels)
                    torch.cuda.empty_cache()

                    loss = classifier_loss / self.classifier_accum_steps
                    loss.backward()
                    train_loss += float(classifier_loss * len(batch[0]))

                    if (count + 1) % self.classifier_accum_steps == 0:
                        optim.step()
                        optim.zero_grad()

                    count = count + 1

                self.model.eval()
                val_loss = 0
                for batch in val_dataloader:
                    with torch.no_grad():
                        val_ids = batch[0].to(rank)
                        attention_mask = batch[1].to(rank)
                        labels = batch[2].to(rank)
                        _, classifier_loss = self.model(val_ids, pred_mode="classifier", attention_mask=attention_mask, labels=labels)
                        val_loss += float(classifier_loss * len(batch[0]))

                print(f"epoch:{epoch}\tloss:{train_loss:0.4f}\tval_loss:{val_loss:0.4f}")

                # 判断early stop是否触发
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    early_stop_count = 0
                    # 保存当前最好的模型
                    self.save_model(file_name="final_model_classifier.pt")
                else:
                    early_stop_count += 1
                    # 判断是否达到早停标准
                    if early_stop_count > 3:
                        print("early stop")
                        break

            # 保存本次bertclassifier到checkpoint供后续测试使用
            check_point_path = os.path.join(self.dataset_dir + f'checkpoints')
            if not os.path.exists(check_point_path):
                os.mkdir(check_point_path)
            save_target = os.path.join(check_point_path, f'classifier_checkpoint_{self_iter}.pt')
            torch.save(self.model, save_target)
            print(f"Classifier checkpoint is saved ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

            # 保存本次bertclassifier
            self.save_model(file_name=self.args.final_model_classifier)
            print(f"Classifier is saved to {self.args.final_model_classifier} ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

            # eval当前模型的eval结果写入文件
            classifier_macro_f1, classifier_micro_f1, _, _ = self.eval_classifier(mode="testset_template")
            f_out = open(out_file, 'a+')
            f_out.write(
                f"selftrain{self_iter},{classifier_macro_f1},{classifier_micro_f1}\n"
                f"Self-training iteration {self_iter}: Macro-F1: {classifier_macro_f1}, Micro-F1: {classifier_micro_f1}\n")
            f_out.close()

            torch.cuda.empty_cache()

    def train_classifier_selftrain_without_template(self):

        out_file = os.path.join(self.dataset_dir, self.args.out_file)
        f_out = open(out_file, 'a+')
        f_out.write(f"Classifier selftraining start {datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Classifier epoch: {self.classifier_epoch}  Selftraining learning rate: {self.classifier_lr} "
                    f"Prompt threshold: {self.args.prompt_thresh}  Classifier threshold: {self.args.classifier_thresh}"
                    f"Selftrain batch: {self.args.classifier_train_batch_size} "
                    f"Classifier accum step: {self.args.classifier_accum_steps} "
                    f"trainset per: {self.args.train_per}\n"
                    f"Raw classifier\n{self.args.prompt_thresh}\n{self.args.classifier_thresh}\n{self.classifier_lr}\n"
                    f"{self.classifier_epoch}\n{self.args.classifier_train_batch_size}\n"
                    f"{self.args.classifier_accum_steps}\n{self.args.train_per}\n\n")
        f_out.close()

        self.train_data = self.create_dataset(self.args.dataset_dir, self.args.train_file, self.args.train_label_file, "train.pt")
        self.test_data = self.create_dataset(self.args.dataset_dir, self.args.test_file, self.args.test_label_file, "test.pt")
        fp_path = os.path.join(self.dataset_dir, "fp")
        self.model = LOTClassModel.from_pretrained(fp_path, output_attentions=False,
                                                   output_hidden_states=False, num_labels=self.num_class)

        # self training start
        for self_iter in range(self.args.self_training_max_time):
            print(f"Self training iteration {self_iter} start...")

            # confidence set comes from promptclassifier at the initial stage
            if self_iter == 0:
                _, _, doc_pred_processed, doc_pred_label, _ = self.write_results_multi_template(
                    out_file=self.args.out_file, template_count=self.args.template_count, mode="trainset_template")

            # confidence set comes from BERTclassifier at the initial stage
            else:
                _, _, doc_pred, doc_pred_label = self.eval_classifier(mode="trainset")
                doc_pred_processed = torch.from_numpy(doc_pred).to(self.device)
                doc_pred_label = torch.from_numpy(doc_pred_label).to(self.device)

            softmaxed_pred_pt = self.softmax(doc_pred_processed)
            confidence = torch.max(softmaxed_pred_pt, dim=1)
            count = Counter(confidence[1].tolist())

            confidence_np = confidence[0].cpu().numpy()

            # find confidence predictions and built train set dataloader and val set dataloader
            if self_iter == 0:
                confidence_ids = torch.tensor(np.where(confidence_np > self.args.prompt_thresh)[0])
            else:
                confidence_ids = torch.tensor(np.where(confidence_np > self.args.classifier_thresh)[0])

            # save evaluation result of confidence set
            pseudo_label_confidence_set = torch.index_select(doc_pred_label.to("cpu"), dim=0, index=confidence_ids).numpy()
            true_label_confidence_set = torch.index_select(self.train_data["labels"], dim=0, index=confidence_ids).numpy()
            confidence_set_macro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='macro')
            confidence_set_micro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='micro')

            # explore the different thresh
            candidate_threshs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
            self.test_thresh(confidence_np=confidence_np, doc_pred_label=doc_pred_label, candidate_threshs=candidate_threshs)

            train_dataloader, val_dataloader, each_cat_count = self.prepare_selftrain_data(self.train_data,
                confidence_ids, doc_pred_label, softmaxed_pred_pt, train_per=self.args.train_per, min_sample_per_class=10, pseudo_label_mode=self.args.pseudo_label_mode)

            f_out = open(out_file, 'a+')
            f_out.write(f"{str(count)}\nConfidence size: {confidence_ids.size()[0]}, Each cat count: {each_cat_count}, Macro-F1: "
                        f"{str(float(confidence_set_macro_f1))}, Micro-F1: {str(float(confidence_set_micro_f1))}\n"
                        f"{confidence_ids.size()[0]}/{each_cat_count},{str(float(confidence_set_macro_f1))},{str(float(confidence_set_micro_f1))}\n")
            f_out.close()

            optim = AdamW(self.model.parameters(), lr=self.classifier_lr)
            rank = self.device

            self.model.to(0)
            self.model.train()

            torch.cuda.empty_cache()

            min_val_loss = 1e10
            early_stop_count = 0

            for epoch in range(self.classifier_epoch):
                train_loss = 0
                count = 0
                for batch in train_dataloader:

                    optim.zero_grad()
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    labels = batch[2].to(rank)
                    _, classifier_loss = self.model(input_ids, pred_mode="raw_classifier", attention_mask=input_mask, labels=labels)
                    torch.cuda.empty_cache()

                    loss = classifier_loss / self.classifier_accum_steps
                    loss.backward()
                    train_loss += float(classifier_loss * len(batch[0]))

                    if (count + 1) % self.classifier_accum_steps == 0:
                        optim.step()
                        optim.zero_grad()
                    # train_loss += float(loss * input_ids.size()[0])

                    count = count + 1

                self.model.eval()
                val_loss = 0
                for batch in val_dataloader:
                    with torch.no_grad():
                        val_ids = batch[0].to(rank)
                        attention_mask = batch[1].to(rank)
                        labels = batch[2].to(rank)
                        _, classifier_loss = self.model(val_ids, pred_mode="raw_classifier", attention_mask=attention_mask, labels=labels)
                        val_loss += float(classifier_loss * len(batch[0]))

                print(f"epoch:{epoch}\tloss:{train_loss:0.4f}\tval_loss:{val_loss:0.4f}")

                # 判断early stop是否触发
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    early_stop_count = 0
                    # 保存当前最好的模型
                    self.save_model(file_name="final_model_classifier.pt")
                else:
                    early_stop_count += 1
                    # 判断是否达到早停标准
                    if early_stop_count > 3:
                        print("early stop")
                        break

            # # 保存本次bertclassifier到checkpoint供后续测试使用
            # check_point_path = os.path.join(self.dataset_dir + f'checkpoints')
            # if not os.path.exists(check_point_path):
            #     os.mkdir(check_point_path)
            # save_target = os.path.join(check_point_path, f'classifier_checkpoint_{self_iter}.pt')
            # torch.save(self.model, save_target)
            # print(f"Classifier checkpoint is saved ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")
            #
            # # 保存本次bertclassifier
            # self.save_model(file_name=self.args.final_model_classifier)
            # print(f"Classifier is saved to {self.args.final_model_classifier} ({datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')})")

            # eval当前模型的eval结果写入文件
            classifier_macro_f1, classifier_micro_f1, _, _ = self.eval_classifier(mode="testset")
            f_out = open(out_file, 'a+')
            f_out.write(
                f"selftrain{self_iter},{classifier_macro_f1},{classifier_micro_f1}\n"
                f"Self-training iteration {self_iter}: Macro-F1: {classifier_macro_f1}, Micro-F1: {classifier_micro_f1}\n")
            f_out.close()

            torch.cuda.empty_cache()

    def supervised_classifier_finetuning(self):
        out_file = os.path.join(self.dataset_dir, self.args.out_file)

        f_out = open(out_file, 'a+')
        f_out.write(f"Classifier selftraining start {datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Classifier epoch: {self.classifier_epoch}  Selftraining learning rate: {self.classifier_lr} "
                    f"Prompt threshold: {self.args.prompt_thresh}  Classifier threshold: {self.args.classifier_thresh}"
                    f"Selftrain batch: {self.args.classifier_train_batch_size} "
                    f"Classifier accum step: {self.args.classifier_accum_steps} "
                    f"trainset per: {self.args.train_per}\n\n")
        f_out.close()

        self.train_data = self.create_dataset(self.args.dataset_dir, self.args.train_file, self.args.train_label_file, "train.pt")
        self.test_data = self.create_dataset(self.args.dataset_dir, self.args.test_file, self.args.test_label_file, "test.pt")
        fp_path = os.path.join(self.dataset_dir, "fp")
        self.model = LOTClassModel.from_pretrained(fp_path, output_attentions=False,
                                                   output_hidden_states=False, num_labels=self.num_class)

        train_data = TensorDataset(self.train_data["input_ids"], self.train_data["attention_masks"], self.train_data["labels"])
        # 划分为训练集和验证集
        train_data_train_size = round(self.train_data["input_ids"].shape[0] * self.args.train_per)
        train_data_val_size = self.train_data["input_ids"].shape[0] - train_data_train_size
        train_data_train, train_data_val = torch.utils.data.random_split(train_data, [train_data_train_size, train_data_val_size])

        train_dataloader = DataLoader(train_data_train, batch_size=self.classifier_train_batch_size, shuffle=True)
        val_dataloader = DataLoader(train_data_val, batch_size=self.classifier_train_batch_size, shuffle=True)

        optim = AdamW(self.model.parameters(), lr=self.classifier_lr)
        rank = self.device

        self.model.to(0)
        self.model.train()

        torch.cuda.empty_cache()

        min_val_loss = 1e10
        early_stop_count = 0

        for epoch in range(self.classifier_epoch):
            train_loss = 0
            count = 0
            for batch in train_dataloader:

                optim.zero_grad()
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                labels = batch[2].to(rank)
                _, classifier_loss = self.model(input_ids, pred_mode="raw_classifier", attention_mask=input_mask,
                                                labels=labels)
                torch.cuda.empty_cache()

                loss = classifier_loss / self.classifier_accum_steps
                loss.backward()
                train_loss += float(classifier_loss * len(batch[0]))

                if (count + 1) % self.classifier_accum_steps == 0:
                    optim.step()
                    optim.zero_grad()

                count = count + 1

            self.model.eval()
            val_loss = 0
            for batch in val_dataloader:
                with torch.no_grad():
                    val_ids = batch[0].to(rank)
                    attention_mask = batch[1].to(rank)
                    labels = batch[2].to(rank)
                    _, classifier_loss = self.model(val_ids, pred_mode="raw_classifier", attention_mask=attention_mask,
                                                    labels=labels)
                    val_loss += float(classifier_loss * len(batch[0]))

            print(f"epoch:{epoch}\tloss:{train_loss:0.4f}\tval_loss:{val_loss:0.4f}")

            # 判断early stop是否触发
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stop_count = 0
                # 保存当前最好的模型
                self.save_model(file_name="final_model_classifier.pt")
            else:
                early_stop_count += 1
                # 判断是否达到早停标准
                if early_stop_count > 5:
                    print("early stop")
                    break

        # eval当前模型的eval结果写入文件
        classifier_macro_f1, classifier_micro_f1, _, _ = self.eval_classifier(mode="testset")
        f_out = open(out_file, 'a+')
        f_out.write(
            f"Supervised learning:  Macro-F1: {classifier_macro_f1}, Micro-F1: {classifier_micro_f1}\n")
        f_out.close()

        torch.cuda.empty_cache()

    def train_classifier_selftrain_multi_template_multi_classifier(self, template_count=1):
        # out_file = os.path.join(self.dataset_dir, self.args.out_file)
        #
        # # self training start
        # for self_iter in range(self.args.self_training_max_time):
        #     print(f"Self training iteration {self_iter} start...")
        #
        #     # confidence set comes from promptclassifier at the initial stage
        #     if self_iter == 0:
        #         _, _, doc_pred, doc_pred_label, _ = self.write_results_multi_template(
        #             out_file=self.args.out_file, template_count=self.args.template_count, mode="trainset_template")
        #         # soft_result_np = self.softmax_np(np.array(doc_pred.to("cpu")))
        #         # soft_result_pt = self.softmax_pt(doc_pred)
        #
        #         # Z-score normalization
        #         mu = torch.mean(doc_pred, dim=1).unsqueeze(1).repeat(1, doc_pred.size()[1])
        #         sigma = torch.std(doc_pred, dim=1).unsqueeze(1).repeat(1, doc_pred.size()[1])
        #         doc_pred_processed = (doc_pred - mu) / sigma
        #
        #     # confidence set comes from BERTclassifier at the initial stage
        #     else:
        #         _, _, doc_pred, doc_pred_label = self.eval_classifier_ensemble(mode="trainset_ensemble", template_count=template_count)
        #         doc_pred_processed = torch.from_numpy(doc_pred).to(self.device)
        #         doc_pred_label = torch.from_numpy(doc_pred_label).to(self.device)
        #
        #     softmaxed_pred_pt = self.softmax(doc_pred_processed)
        #     confidence = torch.max(softmaxed_pred_pt, dim=1)
        #     count = Counter(confidence[1].tolist())
        #
        #     confidence_np = confidence[0].cpu().numpy()
        #
        #     # find confidence predictions and built train set dataloader and val set dataloader
        #     if self_iter == 0:
        #         confidence_ids = torch.tensor(np.where(confidence_np > self.args.prompt_thresh)[0])
        #     else:
        #         confidence_ids = torch.tensor(np.where(confidence_np > self.args.classifier_thresh)[0])
        #
        #     # save evaluation result of confidence set
        #     pseudo_label_confidence_set = torch.index_select(doc_pred_label.to("cpu"), dim=0, index=confidence_ids).numpy()
        #     true_label_confidence_set = torch.index_select(self.train_data_with_template["labels"], dim=0, index=confidence_ids).numpy()
        #     confidence_set_macro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='macro')
        #     confidence_set_micro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='micro')
        #
        #     # explore the different thresh
        #     candidate_threshs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
        #     self.test_thresh(confidence_np=confidence_np, doc_pred_label=doc_pred_label, candidate_threshs=candidate_threshs)
        #
        #     # 依次训练对应五个模板的五个分类器
        #     for template_id in range(template_count):
        #
        #         loader_file = os.path.join(self.dataset_dir, f"final_model_classifier_{template_id}")
        #         self.model.load_state_dict(torch.load(loader_file))
        #         exec(f"self.pre_template = self.args.pre_template_{template_id}")
        #         exec(f"self.post_template = self.args.post_template_{template_id}")
        #         self.train_data_with_template = self.text_fit_template(self.dataset_dir,
        #                                                                self.args.train_file,
        #                                                                self.args.train_label_file,
        #                                                                                  f'train_with_template_{template_id}.pt',
        #                                                                pre_temp=self.pre_template,
        #                                                                post_temp=self.post_template)
        #         self.test_data_with_template = self.text_fit_template(self.dataset_dir,
        #                                                               self.args.test_file,
        #                                                               self.args.test_label_file,
        #                                                                                 f'test_with_template_{template_id}.pt',
        #                                                               pre_temp=self.pre_template,
        #                                                               post_temp=self.post_template)
        #         self.label_name_data_with_template = self.label_name_data_fit_template(self.dataset_dir,
        #                                                                                label_name_loader_name=f'label_name_data_with_pre_post_template_{template_id}.pt',
        #                                                                                train_data_with_template=self.train_data_with_template)
        #
        #         train_dataloader, val_dataloader, each_cat_count = self.prepare_selftrain_data(self.train_data_with_template,
        #                                                                                        confidence_ids, doc_pred_label, softmaxed_pred_pt, train_per=0.8, min_sample_per_class=10, pseudo_label_mode=self.args.pseudo_label_mode)
        #
        #         f_out = open(out_file, 'a+')
        #         f_out.write(f"{str(count)}\nConfidence size: {confidence_ids.size()[0]}, Each cat count: {each_cat_count}, Macro-F1: "
        #                     f"{str(float(confidence_set_macro_f1))}, Micro-F1: {str(float(confidence_set_micro_f1))}\n")
        #         f_out.close()
        #
        #         optim = AdamW(self.model.parameters(), lr=self.classifier_lr)
        #         rank = self.device
        #
        #         self.model.to(0)
        #         self.model.train()
        #
        #         torch.cuda.empty_cache()
        #
        #         min_val_loss = 1e10
        #         early_stop_count = 0
        #
        #         for epoch in range(self.classifier_epoch):
        #             train_loss = 0
        #             count = 0
        #             for batch in train_dataloader:
        #
        #                 optim.zero_grad()
        #                 input_ids = batch[0].to(rank)
        #                 input_mask = batch[1].to(rank)
        #                 labels = batch[2].to(rank)
        #                 _, classifier_loss = self.model(input_ids, pred_mode="classifier", attention_mask=input_mask, labels=labels)
        #                 torch.cuda.empty_cache()
        #
        #                 loss = classifier_loss / self.classifier_accum_steps
        #                 loss.backward()
        #                 train_loss += float(classifier_loss * len(batch[0]))
        #
        #                 if (count + 1) % self.classifier_accum_steps == 0:
        #                     optim.step()
        #                     optim.zero_grad()
        #
        #                 count = count + 1
        #
        #             self.model.eval()
        #             val_loss = 0
        #             for batch in val_dataloader:
        #                 with torch.no_grad():
        #                     val_ids = batch[0].to(rank)
        #                     attention_mask = batch[1].to(rank)
        #                     labels = batch[2].to(rank)
        #                     _, classifier_loss = self.model(val_ids, pred_mode="classifier", attention_mask=attention_mask, labels=labels)
        #                     val_loss += float(classifier_loss * len(batch[0]))
        #
        #             print(f"epoch:{epoch}\tloss:{train_loss:0.4f}\tval_loss:{val_loss:0.4f}")
        #
        #             # 判断early stop是否触发
        #             if val_loss < min_val_loss:
        #                 min_val_loss = val_loss
        #                 early_stop_count = 0
        #                 # 保存当前最好的模型
        #                 self.save_model(file_name=f"final_model_classifier_{template_id}.pt")
        #             else:
        #                 early_stop_count += 1
        #                 # 判断是否达到早停标准
        #                 if early_stop_count > 5:
        #                     print("early stop")
        #                     break
        #
        #         # eval当前模型的eval结果写入文件
        #         classifier_macro_f1, classifier_micro_f1, _, _ = self.eval_classifier(mode="testset_template")
        #         f_out = open(out_file, 'a+')
        #         f_out.write(
        #             f"Self-training iteration {self_iter}: Macro-F1: {classifier_macro_f1}, Micro-F1: {classifier_micro_f1}\n")
        #         f_out.close()
        torch.cuda.empty_cache()

    def prepare_selftrain_data(self, source_data, confidence_ids, doc_pred_label, softmaxed_pred_pt, train_per=0.8, min_sample_per_class=10, pseudo_label_mode="hard"):

        # check each category
        softmaxed_pred_np = softmaxed_pred_pt.cpu().numpy()
        doc_pred_label_np = doc_pred_label.cpu().numpy()
        confidence_labels = doc_pred_label_np[confidence_ids.cpu().numpy()]
        count = Counter(list(confidence_labels.tolist()))

        # prepare each category ids for sampling
        candidate_ids_train = []
        candidate_ids_val = []
        for category_id in range(len(self.label_name_dict.keys())):
            # seperate train and val ids (mutual exclusive)
            if count[category_id] < min_sample_per_class:
                candidate_ids = softmaxed_pred_np[:, category_id].argsort()[-min_sample_per_class:][::-1]

            elif count[category_id] >= min_sample_per_class:
                candidate_ids = confidence_ids[np.where(confidence_labels == category_id)]

            candidate_ids_count_train = round(len(candidate_ids) * train_per)
            candidate_ids_train_i = np.random.choice(candidate_ids, candidate_ids_count_train, replace=False)
            candidate_ids_val_i = np.array([i for i in candidate_ids.tolist() if i not in list(candidate_ids_train_i)], dtype='int64')
            candidate_ids_train.append(candidate_ids_train_i)
            candidate_ids_val.append(candidate_ids_val_i)

        # build train and val ids
        confidence_ids_train = []
        confidence_ids_val = []
        each_cat_count_train = round(np.median([len(i) for i in candidate_ids_train]))
        each_cat_count_val = round(np.median([len(i) for i in candidate_ids_val]))
        for i in candidate_ids_train:
            confidence_ids_train_i = np.random.choice(i, each_cat_count_train, replace=True)
            confidence_ids_train += list(confidence_ids_train_i)
        for i in candidate_ids_val:
            confidence_ids_val_i = np.random.choice(i, each_cat_count_val, replace=True)
            confidence_ids_val += list(confidence_ids_val_i)

        confidence_ids_train_ts = torch.tensor(confidence_ids_train).to(self.device)
        confidence_ids_val_ts = torch.tensor(confidence_ids_val).to(self.device)

        # pseudo label
        if pseudo_label_mode == "soft":
            confidcence_pred_scores_train = torch.index_select(softmaxed_pred_pt.to(self.device), dim=0, index=confidence_ids_train_ts)
            weight_train = confidcence_pred_scores_train ** 2 / torch.sum(confidcence_pred_scores_train, dim=0)
            pseudo_label_train = (weight_train.t() / torch.sum(weight_train, dim=1)).t()

            confidcence_pred_scores_val = torch.index_select(softmaxed_pred_pt.to(self.device), dim=0, index=confidence_ids_val_ts)
            weight_val = confidcence_pred_scores_val ** 2 / torch.sum(confidcence_pred_scores_val, dim=0)
            pseudo_label_val = (weight_val.t() / torch.sum(weight_val, dim=1)).t()

        elif pseudo_label_mode == "hard":
            pseudo_label_train_index = torch.index_select(doc_pred_label.to(self.device), dim=0, index=confidence_ids_train_ts)
            pseudo_label_val_index = torch.index_select(doc_pred_label.to(self.device), dim=0, index=confidence_ids_val_ts)
            # pseudo_label_train = torch.zeros(len(confidence_ids_train), len(self.label_name_dict.keys())).to(self.device).scatter_(1, pseudo_label_train_index.unsqueeze(-1), 1)
            # pseudo_label_val = torch.zeros(len(confidence_ids_val), len(self.label_name_dict.keys())).to(self.device).scatter_(1, pseudo_label_val_index.unsqueeze(-1), 1)

        train_data = TensorDataset(torch.index_select(source_data["input_ids"].to(0), dim=0,
                                                      index=confidence_ids_train_ts),
                                   torch.index_select(source_data["attention_masks"].to(0),
                                                      dim=0,
                                                      index=confidence_ids_train_ts), pseudo_label_train_index)

        train_dataloader = DataLoader(train_data, batch_size=self.classifier_train_batch_size, shuffle=True)

        val_data = TensorDataset(torch.index_select(source_data["input_ids"].to(0), dim=0,
                                                    index=confidence_ids_val_ts),
                                 torch.index_select(source_data["attention_masks"].to(0),
                                                    dim=0,
                                                    index=confidence_ids_val_ts), pseudo_label_val_index)

        val_dataloader = DataLoader(val_data, batch_size=self.eval_batch_size, shuffle=True)

        return train_dataloader, val_dataloader, each_cat_count_train

    def eval_classifier(self, mode="testset_template", out_file=None):

        loader_file = os.path.join(self.dataset_dir, self.args.final_model_classifier)
        self.model.load_state_dict(torch.load(loader_file))

        if mode == "testset_template":
            test_data = TensorDataset(self.test_data_with_template["input_ids"],
                                      self.test_data_with_template["attention_masks"],
                                      self.test_data_with_template["labels"])
            dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                                              batch_size=self.eval_batch_size)

        if mode == "trainset_template":
            train_data = TensorDataset(self.train_data_with_template["input_ids"],
                                       self.train_data_with_template["attention_masks"],
                                       self.train_data_with_template["labels"])
            dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data),
                                              batch_size=self.eval_batch_size)

        if mode == "testset":
            test_data = TensorDataset(self.test_data["input_ids"],
                                      self.test_data["attention_masks"],
                                      self.test_data["labels"])
            dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                                              batch_size=self.eval_batch_size)

        if mode == "trainset":
            train_data = TensorDataset(self.train_data["input_ids"],
                                       self.train_data["attention_masks"],
                                       self.train_data["labels"])
            dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data),
                                    batch_size=self.eval_batch_size)

        device = 0
        self.model.eval()
        self.model.to(device)
        iteration = 0
        for batch in dataloader:
            with torch.no_grad():
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                if mode == "trainset_template" or mode == "testset_template":
                    logits = self.model(input_ids, pred_mode="classifier", attention_mask=attention_mask)
                elif mode == "trainset" or mode =="testset":
                    logits = self.model(input_ids, pred_mode="raw_classifier", attention_mask=attention_mask)

                if iteration == 0:
                    y_pred = logits.cpu().numpy()
                    y_pred_label = torch.argmax(logits, dim=1).cpu().numpy()
                    y_true = batch[2].cpu().numpy()
                else:
                    y_pred = np.concatenate((y_pred, logits.cpu().numpy()), axis=0)
                    y_pred_label = np.concatenate((y_pred_label, torch.argmax(logits, dim=1).cpu().numpy()), axis=0)
                    y_true = np.concatenate((y_true, batch[2].cpu().numpy()), axis=0)
                iteration += 1

        macro_f1 = f1_score(y_true, y_pred_label, average='macro')
        micro_f1 = f1_score(y_true, y_pred_label, average='micro')

        if out_file is not None:
            out_file = os.path.join(self.dataset_dir, out_file)
            print(f"The macro f1 is {macro_f1}, micro f1 is {micro_f1}.")
            print(f"Writing prediction results to {out_file}")
            f_out = open(out_file, 'a+')
            f_out.write(f"Classifier result {datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Epoch: {self.classifier_epoch}  Learning rate: {self.classifier_lr}  Train batch: {self.classifier_train_batch_size}  "
                        f"Accum step: {self.classifier_accum_steps}  Train data: {self.test_data_with_template['input_ids'].size()[0]}\n"
                        "Classifier Macro-F1: " + str(float(macro_f1)) + ", Micro-F1: " + str(float(micro_f1)) + "\n\n")

        return macro_f1, micro_f1, y_pred, y_pred_label

    def eval_classifier_ensemble(self, mode="testset", out_file=None, template_count=1):

        y_preds = []
        macro_f1s = []
        micro_f1s = []

        for template_id in range(template_count):

            loader_file = os.path.join(self.dataset_dir, f"final_model_classifier_{template_id}")
            self.model.load_state_dict(torch.load(loader_file))
            exec(f"self.pre_template = self.args.pre_template_{template_id}")
            exec(f"self.post_template = self.args.post_template_{template_id}")
            self.train_data_with_template = self.text_fit_template(self.dataset_dir,
                                                                   self.args.train_file,
                                                                   self.args.train_label_file,
                                                                                     f'train_with_template_{template_id}.pt',
                                                                   pre_temp=self.pre_template,
                                                                   post_temp=self.post_template)
            self.test_data_with_template = self.text_fit_template(self.dataset_dir,
                                                                  self.args.test_file,
                                                                  self.args.test_label_file,
                                                                                    f'test_with_template_{template_id}.pt',
                                                                  pre_temp=self.pre_template,
                                                                  post_temp=self.post_template)
            self.label_name_data_with_template = self.label_name_data_fit_template(self.dataset_dir,
                                                                                   label_name_loader_name=f'label_name_data_with_pre_post_template_{template_id}.pt',
                                                                                   train_data_with_template=self.train_data_with_template)

            if mode == "testset":
                test_data = TensorDataset(self.test_data_with_template["input_ids"],
                                          self.test_data_with_template["attention_masks"],
                                          self.test_data_with_template["labels"])
                dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                                                  batch_size=self.eval_batch_size)

            if mode == "trainset":
                train_data = TensorDataset(self.train_data_with_template["input_ids"],
                                           self.train_data_with_template["attention_masks"],
                                           self.train_data_with_template["labels"])
                dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data),
                                                  batch_size=self.eval_batch_size)

            device = 0
            self.model.eval()
            self.model.to(device)
            iteration = 0
            for batch in dataloader:
                with torch.no_grad():
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    logits = self.model(input_ids, pred_mode="classifier", attention_mask=attention_mask)

                    if iteration == 0:
                        y_pred = logits.cpu().numpy()
                        y_pred_label = torch.argmax(logits, dim=1).cpu().numpy()
                        y_true = batch[2].cpu().numpy()
                    else:
                        y_pred = np.concatenate((y_pred, logits.cpu().numpy()), axis=0)
                        y_pred_label = np.concatenate((y_pred_label, torch.argmax(logits, dim=1).cpu().numpy()), axis=0)
                        y_true = np.concatenate((y_true, batch[2].cpu().numpy()), axis=0)
                    iteration += 1

            macro_f1 = f1_score(y_true, y_pred_label, average='macro')
            micro_f1 = f1_score(y_true, y_pred_label, average='micro')

            y_preds.append(y_pred)
            macro_f1s.append(macro_f1)
            micro_f1s.append(micro_f1)

        y_pred_ensemble = torch.mean(y_preds)
        y_pred_label_ensemble = torch.argmax(y_pred_ensemble, dim=1).cpu().numpy()
        macro_f1_ensemble = f1_score(y_true, y_pred_label_ensemble, average='macro')
        micro_f1_ensemble = f1_score(y_true, y_pred_label_ensemble, average='micro')

        if out_file is not None:
            out_file = os.path.join(self.dataset_dir, out_file)
            print(f"The macro f1 is {macro_f1}, micro f1 is {micro_f1}.")
            print(f"Writing prediction results to {out_file}")
            f_out = open(out_file, 'a+')
            f_out.write(f"Classifier result {datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Epoch: {self.classifier_epoch}  Learning rate: {self.classifier_lr}  Train batch: {self.classifier_train_batch_size}  "
                        f"Accum step: {self.classifier_accum_steps}  Train data: {self.test_data_with_template['input_ids'].size()[0]}\n"
                        "Classifier Macro-F1: " + str(float(macro_f1_ensemble)) + ", Micro-F1: " + str(float(micro_f1_ensemble)) + "\n\n")

        return macro_f1_ensemble, micro_f1_ensemble, y_pred_ensemble, y_pred_label_ensemble

    def test_thresh(self, confidence_np, doc_pred_label, candidate_threshs):
        for thresh in candidate_threshs:
            confidence_ids = torch.tensor(np.where(confidence_np > thresh)[0])

            pseudo_label_confidence_set = torch.index_select(doc_pred_label.to("cpu"), dim=0, index=confidence_ids).numpy()
            true_label_confidence_set = torch.index_select(self.train_data_with_template["labels"], dim=0, index=confidence_ids).numpy()
            confidence_set_macro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='macro')
            confidence_set_micro_f1 = f1_score(true_label_confidence_set, pseudo_label_confidence_set, average='micro')

            count = Counter(pseudo_label_confidence_set.tolist())

            out_file = os.path.join(self.dataset_dir, self.args.out_file)
            f_out = open(out_file, 'a+')
            f_out.write(f"Threshold: {thresh},{str(count)},Confidence size: {confidence_ids.size()[0]},Macro-F1: "
                        f"{str(float(confidence_set_macro_f1))},Micro-F1: {str(float(confidence_set_micro_f1))}\n")
            f_out.close()
