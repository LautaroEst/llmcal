
import os
from typing import Optional, Dict
from llmcal.data.datasets import SUPPORTED_DATASETS
from datasets import Dataset, load_from_disk
import numpy as np

def load_dataset(dataset_name: str) -> Dict[str,Dataset]:
    try:
        dataset_attrs = SUPPORTED_DATASETS[dataset_name]
    except KeyError:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are {list(SUPPORTED_DATASETS.keys())}")
    full_dataset = {}
    for split_name in ["train", "validation", "test"]:
        full_dataset[split_name] = dataset_attrs['loading_function'](split_name)
    return full_dataset


def main(
    *datasets
):
    
    # Prepare data directory
    data_dir = f"data/"
    os.makedirs(data_dir, exist_ok=True)

    for dataset_name in datasets:
        if os.path.exists(os.path.join(data_dir, dataset_name)):
            print(f"Dataset {dataset_name} already exists in {data_dir}. Skipping.")
            continue
        dataset = load_dataset(dataset=dataset_name)
        for split in dataset:
            dataset[split].save_to_disk(os.path.join(data_dir, dataset_name, split))


if __name__ == "__main__":
    import fire
    fire.Fire(main)