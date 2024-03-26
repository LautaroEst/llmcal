

from collections import defaultdict
import json
from datasets import Dataset
import numpy as np
import pandas as pd

def load_cb(split):
    
    def _load_data(filename):
        data = defaultdict(list)
        with open(filename, "r") as f:
            for line in f:
                myjson = json.loads(line)
                data["idx"].append(myjson['idx'])
                data["hypothesis"].append(myjson['hypothesis'])
                data["premise"].append(myjson['premise'])
                curr_label = myjson['label']
                if curr_label == 'contradiction':
                    data["target"].append(0)
                elif curr_label == 'neutral':
                    data["target"].append(1)
                elif curr_label == 'entailment':
                    data["target"].append(2)
        return data
        
    if split == "test":
        data = _load_data(f"./data/raw_data/tony_zhao/cb/val.jsonl")
    elif split in ["train", "validation"]:
        data = _load_data(f"./data/raw_data/tony_zhao/cb/train.jsonl")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(data["idx"]))
        if split == "validation":
            idx = idx[:50]
        elif split == "train":
            idx = idx[50:]
        data = {k: [v[i] for i in idx] for k, v in data.items()}
    
    dataset = Dataset.from_dict(data)
    dataset = dataset.map(lambda x: {"input": {"premise": x["premise"], "hypothesis": x["hypothesis"]}})
    dataset = dataset.remove_columns(["premise", "hypothesis"])
    return dataset

def load_rte(split):

    def _load_data(filename):
        data = defaultdict(list)
        with open(filename, "r") as f:
            for line in f:
                myjson = json.loads(line)
                data["idx"].append(myjson['idx'])
                data["hypothesis"].append(myjson['hypothesis'])
                data["premise"].append(myjson['premise'])
                curr_label = myjson['label']
                if curr_label == 'not_entailment':
                    data["target"].append(0)
                elif curr_label == 'entailment':
                    data["target"].append(1)
        return data
    
    if split == "test":
        data = _load_data(f"./data/raw_data/tony_zhao/rte/val.jsonl")
    elif split in ["train", "validation"]:
        data = _load_data(f"./data/raw_data/tony_zhao/rte/train.jsonl")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(data["idx"]))
        if split == "validation":
            idx = idx[:50]
        elif split == "train":
            idx = idx[50:]
        data = {k: [v[i] for i in idx] for k, v in data.items()}

    dataset = Dataset.from_dict(data)
    dataset = dataset.map(lambda x: {"input": {"premise": x["premise"], "hypothesis": x["hypothesis"]}})
    dataset = dataset.remove_columns(["premise", "hypothesis"])
    return dataset

def load_trec(split):

    def _load_data(filename):
        idx2label = {0: "NUM", 1: "LOC", 2: "HUM", 3: "DESC", 4: "ENTY", 5: "ABBR"}
        inv_label_dict = {v: k for k, v in idx2label.items()}
        sentences = []
        labels = []
        with open(filename, 'r') as train_data:
            for line in train_data:
                label = line.split(' ')[0].split(':')[0]
                label = inv_label_dict[label]
                sentence = ' '.join(line.split(' ')[1:]).strip()
                sentence = sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
                labels.append(label)
                sentences.append(sentence)
        return {
            'idx': list(range(len(sentences))),
            'question': sentences,
            'target': labels,
        }
    
    if split == "test":
        data = _load_data(f"./data/raw_data/tony_zhao/trec/test.txt")
    elif split in ["train", "validation"]:
        data = _load_data(f"./data/raw_data/tony_zhao/trec/train.txt")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(data["idx"]))
        if split == "validation":
            idx = idx[:100]
        elif split == "train":
            idx = idx[100:]
        data = {k: [v[i] for i in idx] for k, v in data.items()}

    dataset = Dataset.from_dict(data)
    dataset = dataset.map(lambda x: {"input": {"question": x["question"]}})
    dataset = dataset.remove_columns(["question"])
    return dataset

def load_sst2(split):

    def _load_data(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return {
            'idx': list(range(len(sentences))),
            'sentence': sentences,
            'target': labels,
        }
    
    split = "dev" if split == "validation" else split
    data = _load_data(f"./data/raw_data/tony_zhao/sst2/stsa.binary.{split}")

    dataset = Dataset.from_dict(data)
    dataset = dataset.map(lambda x: {"input": {"sentence": x["sentence"]}})
    dataset = dataset.remove_columns(["sentence"])
    return dataset

def load_agnews(split):

    def _load_data(filename):
        data = pd.read_csv(filename)
        articles = data['Title'] + ". " + data['Description']
        articles = list(
            [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
            in articles]) # some basic cleaning
        labels = list(data['Class Index'])
        labels = [l - 1 for l in labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
        return {
            'idx': list(range(len(articles))),
            'article': articles,
            'target': labels,
        }

    if split == "test":
        data = _load_data(f"./data/raw_data/tony_zhao/agnews/test.csv")
        rs = np.random.RandomState(0)
        idx = rs.permutation(len(data["idx"]))[:1000]
        data = {k: [v[i] for i in idx] for k, v in data.items()}
    elif split in ["train", "validation"]:
        data = _load_data(f"./data/raw_data/tony_zhao/agnews/train.csv")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(data["idx"]))
        if split == "validation":
            idx = idx[:200]
        elif split == "train":
            idx = idx[200:10200]
        data = {k: [v[i] for i in idx] for k, v in data.items()}
    
    dataset = Dataset.from_dict(data)
    dataset = dataset.map(lambda x: {"input": {"article": x["article"]}})
    dataset = dataset.remove_columns(["article"])
    return dataset


def load_dbpedia(split):

    def _load_data(filename):
        data = pd.read_csv(filename)
        articles = data['Text']
        articles = list([item.replace('""', '"') for item in articles])
        labels = list(data['Class'])
        labels = [l - 1 for l in labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
        return {
            'idx': list(range(len(articles))),
            'article': articles,
            'target': labels,
        }

    if split == "test":
        data = _load_data(f"./data/raw_data/tony_zhao/dbpedia/test.csv")
        rs = np.random.RandomState(0)
        idx = rs.permutation(len(data["idx"]))[:1000]
        data = {k: [v[i] for i in idx] for k, v in data.items()}
    elif split in ["train", "validation"]:
        data = _load_data(f"./data/raw_data/tony_zhao/dbpedia/train_subset.csv")
        rs = np.random.RandomState(78)
        idx = rs.permutation(len(data["idx"]))
        if split == "validation":
            idx = idx[:500]
        elif split == "train":
            idx = idx[500:10500]
        data = {k: [v[i] for i in idx] for k, v in data.items()}

    dataset = Dataset.from_dict(data)
    dataset = dataset.map(lambda x: {"input": {"article": x["article"]}})
    dataset = dataset.remove_columns(["article"])
    return dataset


TONYZHAO_DATASETS = [
    ("tony_zhao_cb", load_cb),
    ("tony_zhao_rte", load_rte),
    ("tony_zhao_trec", load_trec),
    ("tony_zhao_sst2", load_sst2),
    ("tony_zhao_agnews", load_agnews),
    ("tony_zhao_dbpedia", load_dbpedia),
]


