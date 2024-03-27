
from collections import defaultdict
import os
from typing import Optional, List
from datasets import load_from_disk
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from scipy.special import log_softmax


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
    title: str,
    metrics: List[str],
    bootstrap: int = 100,
    random_state: int = 0,
    test: bool = False,
    *experiments: Optional[List[str]]
):
    """
    View the results of an experiment.
    """

    splits = ["train", "validation"]
    if test:
        splits.append("test")

    results = defaultdict(list)
    for experiment in experiments:
        experiment_dir = os.path.join("experiments", experiment)
        for split in splits:
            if not os.path.exists(os.path.join(experiment_dir, split)):
                continue
            split_results = load_from_disk(os.path.join(experiment_dir, split)).flatten().select_columns(["output.logits", "target"]).with_format("numpy")
            logits = split_results["output.logits"]
            labels = split_results["target"]
            for metric in metrics:
                scores = compute_metric(logits, labels, metric, bootstrap = bootstrap, random_state = random_state)
                results["experiment"].extend([experiment] * len(scores))
                results["split"].extend([split] * len(scores))
                results["metric_value"].extend(scores)
                results["metric_name"].extend([metric] * len(scores))
    results = pd.DataFrame(results)
    
    # compute mean and std
    results = results.groupby(["split", "metric_name"]).agg({"metric_value": ["mean", "std"]})
    results.columns = results.columns.map("_".join)
    results["metric_value"] = results[f"metric_value_mean"].apply(lambda x: f"{x:.3f}") + " ± " + results[f"metric_value_std"].apply(lambda x: f"{x:.3f}")
    results = results.drop(columns = [f"metric_value_mean", f"metric_value_std"])
    results = results.pivot_table(index="split", columns="metric_name", values="metric_value", aggfunc="first")
    results = results.reindex(index=[split for split in splits if split in results.index])
    print("=" * len(title))
    print(title)
    print()
    print(results)
    print("=" * len(title))
    print()


if __name__ == "__main__":
    from fire import Fire
    Fire(main)