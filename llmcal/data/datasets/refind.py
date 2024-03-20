

import json
from datasets import Dataset


IDX2LABEL = {
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

ENTITY_TYPES = {
    "ORG": "Organization",
    "GPE": "Geopolitical Entity",
    "PERSON": "Person",
    "GOV_AGY": "Government Agency",
    "UNIV": "University",
    "TITLE": "Title",
    "DATE": "Date",
    "MONEY": "Money",
}

ENTITY_TYPES_TO_SHORT = {
    "Organization": "org",
    "Geopolitical Entity": "gpe",
    "Person": "pers",
    "Government Agency": "gov_agy",
    "University": "univ",
    "Title": "title",
    "Date": "date",
    "Money": "money",
}

def load_refind(split):
    label2idx = {v: k for k, v in IDX2LABEL.items()}
    if split == "validation":
        split = "dev"
    with open(f"./data/raw_data/refind_dataset/public/{split}_refind_official.json", "r") as f:
        data = json.load(f)
    cleaned_data = {
        'idx': [], 
        'input': [], 
        'target': []
    }
    for i, sample in enumerate(data):
        cleaned_data['idx'].append(i)
        cleaned_data['input'].append({
            'sentence': " ".join(sample["token"]),
            'entity_1_token': " ".join(sample["token"][sample["e1_start"]:sample["e1_end"]]),
            'entity_1_type': ENTITY_TYPES[sample["e1_type"]],
            'entity_2_token': " ".join(sample["token"][sample["e2_start"]:sample["e2_end"]]),
            'entity_2_type': ENTITY_TYPES[sample["e2_type"]]
        })
        cleaned_data['target'].append(label2idx[sample["relation"]])
    dataset = Dataset.from_dict(cleaned_data)
    return dataset