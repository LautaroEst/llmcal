
import os
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm

DATASETS = {"sst2": 2, "agnews": 4, "dbpedia": 14, "20newsgroups": 20, "banking77": 77}
TEST_SAMPLES = {"sst2": 400, "agnews": 400, "dbpedia": 700, "20newsgroups": 800, "banking77": 1000}
N_SEEDS = 5
FACTORS = [8, 16, 32, 64, 128, 256, 512]

def main():
    rs = np.random.RandomState(8364)
    for dataset in tqdm(DATASETS):
        num_classes = DATASETS[dataset]
        for factor in FACTORS:

            scale = factor / np.log2(num_classes)
            nearest_power_of_2 = 2 ** np.round(np.log2(scale)) # round to nearest power of 2
            num_samples = int(nearest_power_of_2 * num_classes)

            for seed in range(N_SEEDS):

                os.makedirs(f"lists/{dataset}/size={factor}/seed={seed}", exist_ok=True)
                
                full_trainlist = np.loadtxt(f"../llmcal2/lists/{dataset}/train.txt", dtype=int)
                if Path(f"../llmcal2/lists/{dataset}/train_{num_samples}_0.3_{seed}.txt").exists() and Path(f"../llmcal2/lists/{dataset}/val_{num_samples}_0.3_{seed}.txt").exists():
                    samples_list = np.hstack([
                        np.loadtxt(f"../llmcal2/lists/{dataset}/train_{num_samples}_0.3_{seed}.txt", dtype=int),
                        np.loadtxt(f"../llmcal2/lists/{dataset}/val_{num_samples}_0.3_{seed}.txt", dtype=int),
                    ])
                else:
                    seedrs = np.random.RandomState(2834+seed)
                    idx = seedrs.permutation(full_trainlist)
                    samples_list = idx[:num_samples]
                
                np.savetxt(f"lists/{dataset}/size={factor}/seed={seed}/0.0-0.7.txt", samples_list[:(num_samples-int(num_samples*0.3))], fmt="%d")
                np.savetxt(f"lists/{dataset}/size={factor}/seed={seed}/0.7-1.0.txt", samples_list[(num_samples-int(num_samples*0.3)):], fmt="%d")
                np.savetxt(f"lists/{dataset}/size={factor}/seed={seed}/0.0-0.3.txt", samples_list[:(num_samples-int(num_samples*0.7))], fmt="%d")
                np.savetxt(f"lists/{dataset}/size={factor}/seed={seed}/0.0-1.0.txt", samples_list, fmt="%d")
    
    for dataset in tqdm(DATASETS):
        full_train_list = np.loadtxt(f"../llmcal2/lists/{dataset}/train.txt", dtype=int)
        np.savetxt(f"lists/{dataset}/train.txt", full_train_list, fmt="%d")
        full_test_list = np.loadtxt(f"../llmcal2/lists/{dataset}/test.txt", dtype=int)
        np.savetxt(f"lists/{dataset}/test.txt", full_test_list, fmt="%d")
        partial_test_list = np.loadtxt(f"../llmcal2/lists/{dataset}/test_{TEST_SAMPLES[dataset]}.txt", dtype=int)
        np.savetxt(f"lists/{dataset}/test_{TEST_SAMPLES[dataset]}.txt", partial_test_list, fmt="%d")



if __name__ == "__main__":
    main()