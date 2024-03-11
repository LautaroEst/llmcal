
import os
from llmcal.data.utils import load_dataset

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
        dataset = load_dataset(dataset=dataset_name, load_from_hub=True, random_state=None, train_samples=None, validation_samples=None, test_samples=None)
        for split in dataset:
            dataset[split].save_to_disk(os.path.join(data_dir, dataset_name, split))


if __name__ == "__main__":
    import fire
    fire.Fire(main)