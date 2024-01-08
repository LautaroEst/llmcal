
import pandas as pd
import numpy as np
from .base import BaseDataset
from datasets import Dataset

class TonyZhaoTREC(BaseDataset):

    idx2label = {0: "NUM", 1: "LOC", 2: "HUM", 3: "DESC", 4: "ENTY", 5: "ABBR"}
    features = ["sentence"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        inv_label_dict = {v: k for k, v in self.idx2label.items()}
        sentences = []
        labels = []
        with open(f'./data/tony_zhao/trec/{split}.txt', 'r') as train_data:
            for line in train_data:
                label = line.split(' ')[0].split(':')[0]
                label = inv_label_dict[label]
                sentence = ' '.join(line.split(' ')[1:]).strip()
                sentence = sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
                labels.append(label)
                sentences.append(sentence)
        dataset = Dataset.from_dict({
            'idx': list(range(len(sentences))),
            'sentence': sentences,
            'label': labels,
        })
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)
    

class TonyZhaoSST2(BaseDataset):

    idx2label = {0: "negative", 1: "positive"}
    features = ["sentence"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        with open(f"./data/tony_zhao/sst2/stsa.binary.{split}", "r") as f:
            lines = f.readlines()
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        dataset = Dataset.from_dict({
            'idx': list(range(len(sentences))),
            'sentence': sentences,
            'label': labels,
        })
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)
        

class TonyZhaoAGNEWS(BaseDataset):

    idx2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    features = ["article"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        data = pd.read_csv(f'./data/tony_zhao/agnews/{split}.csv')
        articles = data['Title'] + ". " + data['Description']
        articles = list(
            [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
            in articles]) # some basic cleaning
        labels = list(data['Class Index'])
        labels = [l - 1 for l in labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
        
        ##########################################################
        if split == "test":
            rs = np.random.RandomState(0)
            idx = rs.permutation(len(articles))[:1000]
            articles = [articles[i] for i in idx]
            labels = [labels[i] for i in idx]
        ##########################################################
            
        dataset = Dataset.from_dict({
            'idx': list(range(len(articles))),
            'article': articles,
            'label': labels,
        })
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)


class TonyZhaoDBPEDIA(BaseDataset):

    idx2dict = {0: 'Company', 1: 'School', 2: 'Artist', 3: 'Athlete', 4: 'Politician', 5: 'Transportation', 6: 'Building', 7: 'Nature', 8: 'Village', 9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'Book'}
    features = ["article"]

    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        data = pd.read_csv(f'./data/tony_zhao/dbpedia/{split}_subset.csv')
        articles = data['Text']
        articles = list([item.replace('""', '"') for item in articles])
        labels = list(data['Class'])
        labels = [l - 1 for l in labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...

        ##########################################################
        if split == "test":
            rs = np.random.RandomState(1)
            idx = rs.permutation(len(articles))[:1000]
            articles = [articles[i] for i in idx]
            labels = [labels[i] for i in idx]
        ##########################################################

        dataset = Dataset.from_dict({
            'idx': list(range(len(articles))),
            'article': articles,
            'label': labels,
        })
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)



if __name__ == "__main__":
    dataset = TonyZhaoDBPEDIA("train")
    print(dataset[1000:1005])