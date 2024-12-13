
import os
from pathlib import Path
import shutil
import numpy as np


DATASETS = {
    "sst2": {"name":  "SST-2", "num_classes": 2},
    "agnews": {"name":  "AGNews", "num_classes": 4},
    "dbpedia": {"name":  "DBPedia", "num_classes": 14},
    "20newsgroups": {"name":  "20 Newsgroups", "num_classes": 20},
    "banking77": {"name":  "Banking77", "num_classes": 77},
}

SIZES = [8, 16, 32, 64, 128, 256, 512]

TEST_SIZES = {
    "sst2": 400,
    "agnews": 400,
    "dbpedia": 700,
    "20newsgroups": 800,
    "banking77": 1000,
}

def compute_num_samples(sizes, dataset):
    num_classes = DATASETS[dataset]["num_classes"]
    scale = sizes / np.log2(num_classes)
    nearest_power_of_2 = 2 ** np.round(np.log2(scale))
    num_samples = nearest_power_of_2 * num_classes
    return num_samples.astype(int)


def copy_directory_with_ignore(old_path, new_path):
    """
    Recursively copy contents from old_path to new_path, ignoring files inside directories
    matching the pattern 'list=val_*'.
    """
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for root, dirs, files in os.walk(old_path):
        # Relative path from the root of old_path
        rel_path = os.path.relpath(root, old_path)

        # # Skip directories matching the pattern "list=val_*"
        # if "list=val_" in os.path.basename(root) or "list=train_" in os.path.basename(root):
        #     continue

        # Determine the corresponding path in new_path
        dest_dir = os.path.join(new_path, rel_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            print(f"Created directory {dest_dir}")

        # Copy files from the current directory
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)
            if os.path.exists(dest_file):
                print(f"File {dest_file} already exists. Skipping...")
                continue
            shutil.copy2(src_file, dest_file, follow_symlinks=False)
            print(f"Copied {src_file} to {dest_file}")

def main():
    for dataset in DATASETS:
        for size in SIZES:
            num_samples = compute_num_samples(size, dataset)
            for seed in range(5):
                # old_path = f"../llmcal2/outputs/adaptation/llama3.2-1b/instruct/all/test={dataset}/list=train_{num_samples}_0.0_{seed}"
                # new_path = f"outputs/no_adaptation/llama3.2-1b-instruct/{dataset}/size={size}/seed={seed}/test={dataset}/list=0.0-1.0"
                old_path = f"../llmcal2/outputs/adaptation/llama3.2-1b/instruct/all/test={dataset}/list=test_{TEST_SIZES[dataset]}"
                new_path = f"outputs/no_adaptation/llama3.2-1b-instruct/{dataset}/size=all/seed=all/test={dataset}/list=test_{TEST_SIZES[dataset]}"
                copy_directory_with_ignore(old_path, new_path)



if __name__ == "__main__":
    main()
