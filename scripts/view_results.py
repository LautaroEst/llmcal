
from collections import defaultdict
import os
from typing import Optional, List
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from scipy.special import log_softmax
import torch


def _compute_metric(logits, targets, metric):
    if metric == "accuracy":
        return accuracy_score(targets, logits.argmax(axis=1))
    elif metric == "error_rate":
        return 1 - accuracy_score(targets, logits.argmax(axis=1))
    elif metric == "f1_score":
        return f1_score(targets, logits.argmax(axis=1), average="macro")
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
    metrics: List[str],
    bootstrap: int = 100,
    random_state: int = 0,
    *folds: Optional[List[str]]
):
    """
    View the results of all experiments
    """

    splits = ["train", "validation"]
    results = defaultdict(list)

    datasets = [d for d in os.listdir("experiments") if not d.startswith(".")]
    for dataset in datasets:
        valid_folds = [d for d in os.listdir(os.path.join("experiments", dataset)) if not d.startswith(".")]
        if folds:
            valid_folds = [f for f in valid_folds if f in folds]
        for fold in valid_folds:
            prompts = [d for d in os.listdir(os.path.join("experiments", dataset, fold)) if not d.startswith(".")]
            for prompt in prompts:
                models = [d for d in os.listdir(os.path.join("experiments", dataset, fold, prompt)) if not d.startswith(".")]
                for model in models:
                    methods = [d for d in os.listdir(os.path.join("experiments", dataset, fold, prompt, model)) if not d.startswith(".")]
                    for method in methods:
                        for split in splits:
                            path = os.path.join("experiments", dataset, fold, prompt, model, method, f"{split}--logits--predict.pt")
                            if not os.path.exists(path) or ".old" in path:
                                continue
                            logits = torch.load(os.path.join("experiments", dataset, fold, prompt, model, method, f"{split}--logits--predict.pt"))
                            labels = torch.load(os.path.join("experiments", dataset, fold, prompt, model, method, f"{split}--label--predict.pt"))
                            logits = logits.float().numpy()
                            labels = labels.long().numpy()
                            for metric in metrics:
                                scores = compute_metric(logits, labels, metric, bootstrap = int(bootstrap), random_state = (random_state))
                                results["dataset"].extend([dataset] * len(scores))
                                results["fold"].extend([fold] * len(scores))
                                results["prompt"].extend([prompt] * len(scores))
                                results["model"].extend([model] * len(scores))
                                results["method"].extend([method] * len(scores))
                                results["split"].extend([split] * len(scores))
                                results["metric_value"].extend(scores)
                                results["metric_name"].extend([metric] * len(scores))
    results = pd.DataFrame(results)
    
    # compute mean and std of metric per metric, dataset, prompt, model, method and split
    results = results.groupby(["dataset", "prompt", "model", "method", "split", "metric_name"]).agg(
        mean = pd.NamedAgg(column="metric_value", aggfunc="mean"),
        std = pd.NamedAgg(column="metric_value", aggfunc="std"),
    ).reset_index()

    # Format to {mean} ± {std}
    results["metric_value"] = results["mean"].round(2).astype(str) + " ± " + results["std"].round(2).astype(str)
    results = results.drop(columns=["mean", "std"])

    # Show each metric in a different column
    results = results.pivot_table(
        index=["dataset", "prompt", "model", "method", "split"], 
        columns="metric_name", 
        values="metric_value",
        aggfunc=lambda x: x
    ).reset_index()
    results = results.sort_values(["dataset", "prompt", "model", "split", metrics[0]])

    print(results.loc[:,["dataset","split","method","error_rate","norm_cross_entropy"]])



    


if __name__ == "__main__":
    from fire import Fire
    Fire(main)