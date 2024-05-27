from pathlib import Path
from typing import OrderedDict
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.special import log_softmax
from datasets import load_from_disk

from llmcal.utils import load_yaml

method_short2name = {
    "no_adaptation+no_calibration": "No adaptation",
    "lora+no_calibration": "LoRA",
    "no_adaptation+affine_matrix": "Affine Matrix",
    "no_adaptation+affine_vector": "Affine Vector",
    "no_adaptation+affine_scalar": "Affine Scalar",
    "no_adaptation+temperature_scaling": "Temperature Scaling",
    "no_adaptation+bias_only": "Bias Only",
    "lora+affine_matrix": "LoRA + Affine Matrix",
    "lora+affine_vector": "LoRA + Affine Vector",
    "lora+affine_scalar": "LoRA + Affine Scalar",
    "lora+temperature_scaling": "LoRA + Temperature Scaling",
    "lora+bias_only": "LoRA + Bias Only",
}

model_short2name = {
    "lm_tinyllama": "TinyLLAMA",
    "lm_tinyllama_chat": "TinyLLAMA-Chat",
}

dataset_short2name = {
    "sst2": "SST-2",
    "dbpedia": "DBPedia",
    "agnews": "AGNews",
    "banking77": "Banking77",
    "20newsgroups": "20NewsGroups",
    "medical-abstracts": "Medical Abstracts",
}

sizes_short2name = OrderedDict([
    ("small", "Small"),
    ("medium", "Medium"),
    ("large", "Large"),
])

metrics_short2name = {
    "accuracy": "Accuracy",
    "error_rate": "Error Rate",
    "f1_score": "F1 Score",
    "cross_entropy": "Cross Entropy",
}

def load_results_paths():
    root_results_dir = Path("../experiments")
    results = []
    for dataset in root_results_dir.glob("*"):
        for prompt in dataset.glob("*"):
            for model in prompt.glob("*"):
                for base_method in model.glob("*"):
                    for cal_method in base_method.glob("*"):
                        if cal_method.name == ".cache":
                            continue
                        for path in (cal_method / "predictions").glob("*"):
                            dataset_name, size = dataset.name.split("_")
                            if "basic_" in prompt.name:
                                prompt_name = "Basic"
                            elif "instr_" in prompt.name:
                                prompt_name = "Instruction"
                            else:
                                prompt_name = prompt.name
                            n_shots = load_yaml(f"../configs/prompt/{prompt.name}.yaml")["num_shots"]
                            data_config = load_yaml(f"../configs/dataset/{dataset.name}.yaml")
                            num_samples = data_config["train_samples"] + data_config["val_samples"]
                            model_name = model.name
                            base_method_name = load_yaml(f"../configs/base_method/{base_method.name}.yaml")["method"]
                            cal_method_name = load_yaml(f"../configs/calibration_method/{cal_method.name}.yaml")["method"]
                            split_name = path.name
                            results_path = str(path)
                            results.append({
                                "dataset": dataset_short2name[dataset_name],
                                "size": sizes_short2name[size],
                                "num_samples": num_samples,
                                "prompt": prompt_name,
                                "n_shots": n_shots,
                                "model": model_short2name[model_name],
                                "method": method_short2name[base_method_name + "+" + cal_method_name],
                                "split": split_name,
                                "results": results_path
                            })
    return pd.DataFrame(results)

def _compute_metric(logits, targets, metric):
    if metric == "accuracy":
        return (logits.argmax(axis=1) == targets).mean()
    elif metric == "error_rate":
        return (logits.argmax(axis=1) != targets).mean()
    elif metric == "f1_score":
        return f1_score(targets, logits.argmax(axis=1), average="macro")
    elif metric == "cross_entropy":
        logprobs = log_softmax(logits, axis=1)
        return - np.mean(logprobs[np.arange(len(targets)), targets])
    else:
        raise ValueError(f"Metric {metric} is not supported")


def compute_metric(logits, targets, metric, bootstrap, random_state):

    if metric.startswith("norm_"):
        metric = metric[5:]
        minlength = logits.shape[1]
        num_samples = len(targets)
        counts = np.bincount(targets, minlength=minlength)
        priors = counts / num_samples
        priors = np.tile(priors, (num_samples, 1))
        logpriors = np.log(priors)
        
        if bootstrap == 0:
            return [
                _compute_metric(logits, targets, metric) /
                _compute_metric(logpriors, targets, metric)
            ]
        
        rs = np.random.RandomState(random_state)
        values = []
        for _ in range(bootstrap):
            idx = rs.choice(len(targets), len(targets), replace=True)
            values.append(
                _compute_metric(logits[idx], targets[idx], metric) /
                _compute_metric(logpriors[idx], targets[idx], metric)
            )
        return values
    
    if bootstrap == 0:
        return [_compute_metric(logits, targets, metric)]
    
    rs = np.random.RandomState(random_state)
    values = []
    for _ in range(bootstrap):
        idx = rs.choice(len(targets), len(targets), replace=True)
        values.append(_compute_metric(logits[idx], targets[idx], metric))
        
    return values


def compute_results(metrics, bootstrap, random_state, show_test=False):
    df_results = load_results_paths()
    if not show_test:
        df_results = df_results[df_results["split"] != "test"]
    
    for metric in metrics:
        df_results[metric] = ""
        df_results[f"{metric}:mean"] = np.nan
        df_results[f"{metric}:std"] = np.nan

    for idx, row in tqdm(df_results.iterrows(), total=len(df_results)):
        row_results = load_from_disk(row["results"]).with_format("numpy")
        logits = row_results["logits"].astype(float)
        targets = row_results["label"].astype(int)
        for metric in metrics:
            values = compute_metric(logits, targets, metric, bootstrap, random_state)
            mean = np.mean(values)
            std = np.std(values)
            df_results.loc[idx, metric] = f"{mean:.3f} ± {std:.3f}"
            df_results.loc[idx, f"{metric}:mean"] = mean
            df_results.loc[idx, f"{metric}:std"] = std
    
    df_results.drop(columns=["results"], inplace=True)

    return df_results


def plot_results(df, metrics, width=.8, test=False):

    df = df.copy()
    df = df[df["split"] == "test"] if test else df[df["split"] == "validation"]
    
    models_with_prompts = (df["model"]  + "---" + df["prompt"]).unique()
    datasets = df["dataset"].unique()
    sizes = sizes_short2name.values()
    methods = df["method"].unique()

    for model_with_prompt in models_with_prompts:
        model, prompt = model_with_prompt.split("---")
        fig, ax = plt.subplots(len(metrics),len(datasets), figsize=(14, 5), sharex="col")
        if len(metrics) == 1 and len(datasets) == 1:
            ax = np.array([[ax]])
        elif len(metrics) == 1:
            ax = ax[np.newaxis, :]
        elif len(datasets) == 1:
            ax = ax[:, np.newaxis]
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metrics):
                for m, method in enumerate(methods):
                    x = []
                    means = []
                    stds = []
                    for s, size in enumerate(sizes):
                        mask = \
                            (df["model"] == model) & \
                            (df["prompt"] == prompt) & \
                            (df["dataset"] == dataset) & \
                            (df["size"] == size) & \
                            (df["method"] == method)
                        if mask.sum() == 0:
                            continue
                        mean = df[mask][f"{metric}:mean"].values[0]
                        std = df[mask][f"{metric}:std"].values[0]
                        x.append(s - width / 2 + width / (len(methods) - 1) * m)
                        means.append(mean)
                        stds.append(std)
                    ax[j, i].errorbar(
                        np.array(x),
                        np.array(means), 
                        yerr=np.array(stds), 
                        ls = "dotted",
                        label=method,
                        capsize=5,
                        capthick=1,
                        elinewidth=1,
                        color=f"C{m}"
                    )
                ax[j,i].grid(True)
            

            
            sizes_with_samples_num = []
            for size in sizes:
                mask = (df["size"] == size) & (df["model"] == model) & (df["prompt"] == prompt) & (df["dataset"] == dataset)
                if mask.sum() == 0:
                    sizes_with_samples_num.append(size)
                else:
                    sizes_with_samples_num.append(
                        df[mask]["num_samples"].astype(str).unique()[0]
                    )
            ax[-1, i].set_xticks(range(len(sizes)))
            ax[-1, i].set_xticklabels(sizes_with_samples_num, rotation=45, ha="right")
            ax[-1, i].set_xlim(-width, len(sizes) - (1 - width))
        
        for i, dataset in enumerate(datasets):
            ax[0, i].set_title(dataset)
        for j, metric in enumerate(metrics):
            if "norm_" in metric:
                metric_name = "Normalized\n" + metrics_short2name[metric[5:]]
                for i, dataset in enumerate(datasets):
                    ylim = ax[j, i].get_ylim()
                    ax[j, i].set_ylim(0, max(1.2,ylim[1]))
                    ax[j, i].axhline(1, color='black', linestyle='--', linewidth=1, label="Naive")
            else:
                metric_name = metrics_short2name[metric]
            ax[j, 0].set_ylabel(metric_name)

        fig.suptitle(model)
        fig.text(0.5, 0.0, 'Number of training samples', ha='center')

        handles, labels = [], []

        for a in fig.axes:
            hand, lab = a.get_legend_handles_labels()
            for h, l in zip(hand, lab):
                if l not in labels and l != "Naive":
                    handles.append(h)
                    labels.append(l)

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(methods), fancybox=True, shadow=True)
        fig.tight_layout()

