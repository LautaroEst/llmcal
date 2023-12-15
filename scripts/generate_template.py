
import argparse
from llmcal.data import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    return args

def main():

    # Read command args
    args = parse_args()

    # Prepare dataloaders
    dataset = load_dataset(args.dataset_name, split="train", subsample=args.num_shots, random_state=args.seed, sort_by_length=False)

    print("Dataset: %s" % args.dataset_name)
    for i in range(len(dataset)):
        sample = dataset[i]
        for feature in dataset.features:
            print(f"{feature}: {[sample[feature]]}. label: {sample['label']}")
        print()
    
if __name__ == "__main__":
    main()