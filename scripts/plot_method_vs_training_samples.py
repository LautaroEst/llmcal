

from collections import OrderedDict, defaultdict
import os
from typing import List, Optional

import numpy as np
from llmcal.utils import load_yaml
from datasets import load_from_disk
from scipy.special import log_softmax
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

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

METHODS = OrderedDict([
    ("glue_sst2_inst_0-shot-AB_prompt/tinyllama/all", {"label": "No adaptation", "color": "black", "marker": "o",  "markersize": 5, "alpha": 0.8}),
    ("glue_sst2_inst_0-shot-AB_tinyllama-logits/affine_vector", {"label": "Affine vector", "color": "C0", "marker": "o",  "markersize": 5, "linestyle": "--", "linewidth": 1, "alpha": 1})
])

METRICS = {
    "accuracy": {"label": "Accuracy", "ylims": None},
    "error_rate": {"label": "Error rate", "ylims": None},
    "f1_score": {"label": "F1 score", "ylims": None},
    "norm_cross_entropy": {"label": "Normalized\ncross-entropy", "ylims": None},
}


def main(
    *methods,
    baseline_method: Optional[str] = None,
    metrics: List[str] = ["error_rate", "norm_cross_entropy"],
    bootstrap: int = 100,
    random_state: int = 0,
    split: str = "test", 
    title: str = "Method comparison",
    filename: str = "plot.png"
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
                    all_results["n_samples"].extend([config["splits"]["train_samples"] + config["splits"]["validation_samples"]]*len(value))
                    all_results["fold"].extend([fold]*len(value))
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.groupby(["method", "metric", "n_samples"]).agg({"value": ["mean", "std"]}).reset_index()
        num_samples = df["n_samples"].unique()
    else:
        num_samples = [0]
    fig, ax = plt.subplots(len(metrics), 1, figsize=(len(num_samples) * 6, len(metrics) * 5), sharex=True)
    if len(metrics) == 1:
        ax = np.array([ax])

    for i, metric in enumerate(metrics):
        if all_results:
            if metric == "norm_cross_entropy":
                ax[i].axhline(1, color="black", linestyle="--", linewidth=1)
            dfp = df[df["metric"] == metric].pivot(index="n_samples", columns="method", values=("value", "mean"))
            dfp = dfp.rename(columns=lambda x: METHODS[x]["label"])
            dfp = dfp.reindex([v["label"] for k, v in METHODS.items() if k != baseline_method], axis=1)
            dfp = dfp.sort_index(ascending=True)
            yerr = df[df["metric"] == metric].pivot(index="n_samples", columns="method", values=("value", "std"))
            yerr = yerr.rename(columns=lambda x: METHODS[x]["label"])
            yerr = yerr.reindex([v["label"] for k, v in METHODS.items() if k != baseline_method], axis=1)
            yerr = yerr.sort_index(ascending=True)
            dfp.plot(kind="line", ax=ax[i], yerr=yerr, capsize=5, legend=False)

        if baseline_method:
            results = load_from_disk(os.path.join(experiments_dir, baseline_method, split)).flatten().select_columns(["output.logits", "target"]).with_format("numpy")
            logits = results["output.logits"]
            targets = results["target"]
            value = compute_metric(logits, targets, metric, bootstrap, random_state)
            ax[i].errorbar([-1], np.mean(value), yerr=np.std(value), **METHODS[baseline_method])

        ax[i].grid()
        ax[i].set_ylim(METRICS[metric]["ylims"])
        ax[i].set_ylabel(METRICS[metric]["label"], fontsize=15)

    if all_results:
        ax[-1].set_xticks(num_samples)
        ax[-1].set_xticklabels(dfp.index, fontsize=15, rotation=0)
    ax[-1].set_xlabel("Size of training data", fontsize=15)
    ax[-1].legend(fontsize=15, bbox_to_anchor=(0.5,-0.8), ncol=6, loc="lower center")
    ax[0].set_title(title, fontsize=20)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")

if __name__ == "__main__":
    from fire import Fire
    Fire(main)