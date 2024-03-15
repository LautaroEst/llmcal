
import os
from typing import Optional, Dict

from tqdm import tqdm
from llmcal.data.datasets import SUPPORTED_DATASETS
from datasets import Dataset
import numpy as np

def load_dataset(dataset_name: str) -> Dict[str,Dataset]:
    try:
        dataset_loader_fn = SUPPORTED_DATASETS[dataset_name]
    except KeyError:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are {list(SUPPORTED_DATASETS.keys())}")
    full_dataset = {}
    for split_name in ["train", "validation", "test"]:
        full_dataset[split_name] = dataset_loader_fn(split_name)
    return full_dataset


def main(*datasets):
    
    # Prepare data directory
    data_dir = f"data/"
    os.makedirs(data_dir, exist_ok=True)

    for dataset_name in tqdm(datasets):
        if os.path.exists(os.path.join(data_dir, dataset_name)):
            continue
        dataset = load_dataset(dataset_name)
        for split in dataset:
            dataset[split].save_to_disk(os.path.join(data_dir, dataset_name, split))


if __name__ == "__main__":
    import fire
    fire.Fire(main)