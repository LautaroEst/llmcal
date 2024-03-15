
import os
from datasets import load_from_disk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def _compute_metric(logits, targets, metric):
    if metric == "accuracy":
        return accuracy_score(targets, logits.argmax(axis=1))
    else:
        raise ValueError(f"Metric {metric} is not supported")


def compute_metric(logits, targets, metric, bootstrap, random_state):
    if bootstrap == 0:
        return [_compute_metric(logits, targets, metric)]
    
    rs = np.random.RandomState(random_state)
    values = []
    for _ in range(bootstrap):
        idx = rs.choice(len(targets), len(targets), replace=True)
        values.append(_compute_metric(logits[idx], targets[idx], metric))
    return values



def main(
    model: str, # TODO: change for all models available
    train_task: str,
    test_task: str,
    metrics: str,
    bootstrap: int,
    random_state: int,
):
    experiments_dir = f"experiments/{model}/{train_task}/{test_task}"
    metrics = metrics.split(",")

    for split in ["train", "validation", "test"]:
        split_results = {"metric": [], "value": []}
        for fold in os.listdir(experiments_dir):
            fold_dir = os.path.join(experiments_dir, fold)
            if not os.path.isdir(fold_dir):
                continue
            results = load_from_disk(os.path.join(fold_dir, split)).with_format("numpy")
            logits = results["output"]
            targets = results["target"]
            for metric in metrics:
                value = compute_metric(logits, targets, metric, bootstrap, random_state)
                split_results["value"].extend(value)
                split_results["metric"].extend([metric]*len(value))
        
        split_results = pd.DataFrame(split_results)
        print(f"Split: {split}")
        for metric in metrics:
            print(f"{metric}:")
            print(split_results.groupby("metric").get_group(metric).describe())
        print()


if __name__ == "__main__":
    from fire import Fire
    Fire(main)


