
from datasets import load_dataset as load_hf_dataset


def load_dataset(dataset_name, split):
    if dataset_name == "glue/sst2":
        dataset = load_hf_dataset("glue", "sst2", split=split)
    return dataset