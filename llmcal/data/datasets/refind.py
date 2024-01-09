
import json
from datasets import Dataset

from .base import BaseDataset

class REFinD(BaseDataset):
    
    idx2label = {
        0: "no_relation",
        1: "org:date:formed_on",
        2: "org:gpe:operations_in",
        3: "pers:org:member_of",
        4: "pers:org:employee_of",
        5: "pers:gov_agy:member_of",
        6: "org:org:acquired_by",
        7: "org:money:loss_of",
        8: "org:gpe:headquartered_in",
        9: "pers:univ:employee_of",
        10: "org:date:acquired_on",
        11: "pers:univ:attended",
        12: "org:gpe:formed_in",
        13: "org:money:profit_of",
        14: "org:money:cost_of",
        15: "org:org:subsidiary_of",
        16: "org:org:shares_of",
        17: "pers:org:founder_of",
        18: "pers:title:title",
        19: "org:money:revenue_of",
        20: "org:org:agreement_with",
        21: "pers:univ:member_of",
    }
    features = ['sentence', 'entity_1_token', 'entity_1_type', 'entity_2_token', 'entity_2_type']

    entity_types = {
        "ORG": "Organization",
        "GPE": "Geopolitical Entity",
        "PERSON": "Person",
        "GOV_AGY": "Government Agency",
        "UNIV": "University",
        "TITLE": "Title",
        "DATE": "Date",
        "MONEY": "Money",
    }


    def __init__(self, split, subsample=None, random_state=None, sort_by_length=False):
        label2idx = {v: k for k, v in self.idx2label.items()}
        with open(f"./data/refind_dataset/public/{split}_refind_official.json", "r") as f:
            data = json.load(f)
        cleaned_data = {
            'idx': [], 'sentence': [], 'entity_1_token': [], 'entity_1_type': [], 
            'entity_2_token': [], 'entity_2_type': [], 'label': []
        }
        for i, sample in enumerate(data):
            cleaned_data['idx'].append(i)
            cleaned_data['sentence'].append(" ".join(sample["token"]))
            cleaned_data['entity_1_token'].append(" ".join(sample["token"][sample["e1_start"]:sample["e1_end"]]))
            cleaned_data['entity_1_type'].append(self.entity_types[sample["e1_type"]])
            cleaned_data['entity_2_token'].append(" ".join(sample["token"][sample["e2_start"]:sample["e2_end"]]))
            cleaned_data['entity_2_type'].append(self.entity_types[sample["e2_type"]])
            cleaned_data['label'].append(label2idx[sample["relation"]])
        dataset = Dataset.from_dict(cleaned_data)
        super().__init__(dataset, subsample=subsample, random_state=random_state, sort_by_length=sort_by_length)



if __name__ == "__main__":
    dataset = REFinD('train')