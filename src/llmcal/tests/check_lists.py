
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm

DATASETS = {"sst2": 2, "agnews": 4, "dbpedia": 14, "20newsgroups": 20, "banking77": 77}
N_SHOTS = [0, 1, 2, 4, 8, 16, 32, 64]
N_SEEDS = 5
FACTORS = [8, 32, 64, 128, 256, 512]
VAL_PROPS = [0.0, 0.3]
TEST_SAMPLES = {"sst2": 400, "agnews": 400, "dbpedia": 700, "20newsgroups": 800, "banking77": 1000}

def main():
    for dataset in tqdm(DATASETS):
        num_classes = DATASETS[dataset]
        for factor in FACTORS:
            scale = factor / np.log2(num_classes)
            nearest_power_of_2 = 2 ** np.round(np.log2(scale)) # round to nearest power of 2
            num_samples = int(nearest_power_of_2 * num_classes)

            # Read data
            data = pd.read_csv(f"data/{dataset}/all.csv")
            
            # check train, test and test_nsamples lists are ok
            train_list = np.loadtxt(f"lists/{dataset}/train.txt")
            assert data.index.isin(train_list).sum() == len(train_list) and np.unique(train_list).size == len(train_list)

            test_list = np.loadtxt(f"lists/{dataset}/test.txt")
            assert data.index.isin(test_list).sum() == len(test_list) and np.unique(test_list).size == len(test_list)

            test_nsamples_list = np.loadtxt(f"lists/{dataset}/test_{TEST_SAMPLES[dataset]}.txt")
            assert data.index.isin(test_nsamples_list).sum() == len(test_nsamples_list) and np.unique(test_nsamples_list).size == len(test_nsamples_list)

            # Check no overlap between train and test, and train and test_nsamples
            assert len(np.intersect1d(train_list, test_list)) == 0
            assert len(np.intersect1d(train_list, test_nsamples_list)) == 0

            for valprop in VAL_PROPS:
                for seed in range(N_SEEDS):
                    with open(f"lists/{dataset}/size={factor}/valprop={valprop}/seed={seed}/matched.yaml", 'r') as file:
                        matched = yaml.load(file, Loader=yaml.FullLoader)

                    val_size = int(valprop * num_samples)
                    train_size = num_samples - val_size
                    assert len(matched["train"][dataset]) == train_size
                    if val_size > 0:
                        assert len(matched["val"][dataset]) == val_size
                        assert not np.isin(matched["val"][dataset], matched["train"][dataset]).any()
                        assert not np.isin(matched["val"][dataset], test_list).any()
                    else:
                        assert np.isin(matched["val"][dataset], matched["train"][dataset]).all()

                    with open(f"lists/{dataset}/size={factor}/valprop={valprop}/seed={seed}/mismatched.yaml", 'r') as file:
                        mismatched = yaml.load(file, Loader=yaml.FullLoader)

                    assert all([train_dataset != dataset for train_dataset in mismatched["train"]])

                        
    print("All lists are ok!")

if __name__ == '__main__':
    main()