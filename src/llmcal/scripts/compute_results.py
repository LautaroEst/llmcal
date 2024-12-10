
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..src.evaluation.metrics import compute_psr_with_mincal


def read_results(root_results_dir: Path):
    data = []
    for dataset_dir in root_results_dir.iterdir():
        train_dataset = dataset_dir.name
        for size_dir in dataset_dir.iterdir():
            size = int(size_dir.name.split("=")[1])
            for seed_dir in size_dir.iterdir():
                seed = int(seed_dir.name.split("=")[1])
                for method in seed_dir.iterdir():
                    method_name = method.name
                    for train_lst in method.iterdir():
                        train_lst_name = train_lst.name
                        for val_lst in train_lst.iterdir():
                            val_lst_name = val_lst.name
                            for test_dataset_dir in val_lst.iterdir():
                                if not test_dataset_dir.name.startswith("test="):
                                    continue
                                test_dataset = test_dataset_dir.name.split("=")[1]
                                for test_lst in test_dataset_dir.iterdir():
                                    if not test_lst.name.startswith("list="):
                                        continue
                                    test_lst_name = test_lst.name.split("=")[1]
                                    if not (logits_path := test_lst / "logits.csv").exists():
                                        continue
                                    if not (labels_path := test_lst / "labels.csv").exists():
                                        continue
                                    data.append({
                                        "train_dataset": train_dataset,
                                        "size": size,
                                        "seed": seed,
                                        "method": method_name,
                                        "train_lst": train_lst_name,
                                        "val_lst": val_lst_name,
                                        "test_dataset": test_dataset,
                                        "test_lst": test_lst_name,
                                        "logits": logits_path,
                                        "labels": labels_path,
                                    })

    return pd.DataFrame(data)
                        

def compute_metrics(data, metric):
    data_with_metrics = data.copy()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        logits = pd.read_csv(row["logits"], index_col=0, header=None).values.astype(float)
        labels = pd.read_csv(row["labels"], index_col=0, header=None).values.flatten().astype(int)
        if row["test_lst"].startswith("test_"):
            value, min_value = compute_psr_with_mincal(logits, labels, metric, "none")
        else:
            value, min_value = 0., 0.
        data_with_metrics.loc[i, "result"] = value
        data_with_metrics.loc[i, "min_result"] = min_value
    data_with_metrics = data_with_metrics.drop(columns=["logits", "labels"])
    return data_with_metrics


def main(
    metric: str,
    root_results_dir: str,
    output_path: str,
):
    # Read results
    root_results_dir = Path(root_results_dir)
    data = read_results(root_results_dir)
    
    # Compute metrics
    data_with_metrics = compute_metrics(data, metric)
    data_with_metrics.to_json(output_path, orient="records", lines=True)
    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)