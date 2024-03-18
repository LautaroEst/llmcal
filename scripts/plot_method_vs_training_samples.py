

from collections import defaultdict
import os
from typing import List

import numpy as np
from llmcal.utils import load_yaml
from datasets import load_from_disk
from scipy.special import log_softmax
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _compute_metric(logits, targets, metric):
    if metric == "accuracy":
        return accuracy_score(targets, logits.argmax(axis=1))
    elif metric == "error_rate":
        return 1 - accuracy_score(targets, logits.argmax(axis=1))
    elif metric == "norm_cross_entropy":
        naive_priors = np.bincount(targets, minlength=logits.shape[1]) / len(targets)
        naive_entropy = - np.mean(np.log(naive_priors[targets]))
        logprobs = log_softmax(logits, axis=1)
        cross_entropy = - np.mean(logprobs[np.arange(len(targets)), targets])
        return cross_entropy / naive_entropy
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
    *methods,
    metrics: List[str] = ["error_rate", "norm_cross_entropy"],
    bootstrap: int = 100,
    random_state: int = 0,
    split: str = "test", 
):
    experiments_dir = f"experiments"
    all_results = defaultdict(list)
    for task in os.listdir(experiments_dir):
        for model in os.listdir(os.path.join(experiments_dir,task)):
            method = os.path.join(task, model)
            if method not in methods:
                continue
            for fold in os.listdir(os.path.join(experiments_dir,task,model)):
                config = load_yaml(os.path.join(experiments_dir, task, model, fold, "config.yaml"))
                results = load_from_disk(os.path.join(experiments_dir, task, model, fold, split)).flatten().select_columns(["output.logits", "target"]).with_format("numpy")
                logits = results["output.logits"]
                targets = results["target"]
                for metric in metrics:
                    value = compute_metric(logits, targets, metric, bootstrap, random_state)
                    all_results["value"].extend(value)
                    all_results["metric"].extend([metric]*len(value))
                    all_results["method"].extend([method]*len(value))
                    all_results["n_samples"].extend([config["splits"]["train_samples"]]*len(value))
                    all_results["fold"].extend([fold]*len(value))
    df = pd.DataFrame(all_results)
    import pdb; pdb.set_trace()
    print(df)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)