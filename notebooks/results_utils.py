import torch
from copy import deepcopy
from pathlib import Path
from typing import OrderedDict
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score as _f1_score
from scipy.special import log_softmax, softmax
from datasets import load_from_disk

from llmcal.utils import load_yaml
from affinecal import cal_loss, min_cal

dataset_short2name = OrderedDict([
    ("sst2", {"name": "SST-2", "num_classes": 2}),
    ("agnews", {"name": "AG News", "num_classes": 4}),
    # ("medical-abstracts", {"name": "Medical Abstracts", "num_classes": 5}),
    ("dbpedia", {"name": "DBpedia", "num_classes": 14}),
    ("20newsgroups", {"name": "20 Newsgroups", "num_classes": 20}),
    ("banking77", {"name": "Banking77", "num_classes": 77}),
])

metrics_short2name = {
    "accuracy": "Acc",
    "error_rate": "ER",
    "f1_score": "F1",
    "cross_entropy": "CE",
    "ece": "ECE",
    "cal_loss_bias": "CalLoss(%)",
    "cal_loss_nobias": "CalLoss(%)\nWithout bias",
    "min_calibration_bias": "CE (PostCal)",
    "min_calibration_nobias": "Normalized CE after\nBias Calibration",
}
marker_size = 7
supported_methods = OrderedDict([
    
    ("temp_scaling", {"label": "Scale-only Calibration\n(Temperature Scaling)", "color": "tab:orange", "ls": "--", "marker": "*","ms": marker_size}),
    ("bias_only", {"label": "Bias-only Calibration", "color": "tab:green", "ls": "--", "marker": "*","ms": marker_size}),    
    ("affine_scalar", {"label": "DP Calibration", "color": "tab:blue", "ls": "--", "marker": "*", "ms": marker_size}),
    ("lora", {"label": "LoRA", "color": "tab:red", "ls": "--", "marker": "*","ms": marker_size}),
    # ("lora+affine_scalar", {"label": "LoRA + Affine Calibration", "color": "tab:blue", "ls": "--"}),
    # ("lora+temp_scaling", {"label": "LoRA + Scale Only Calibration", "color": "tab:orange", "ls": "--"}),
    # ("lora+bias_only", {"label": "LoRA + Bias Only Calibration", "color": "tab:green", "ls": "--"}),
    # ("affine_scalar_no_es", {"label": "Affine Calibration\n(NO Early Stopping)", "color": "tab:blue", "ls": "-."}),
    ("lora+affine_scalar_train_on_val", {"label": "Lora + DP Calibration", "color": "tab:purple", "marker": "*", "alpha": 1., "ls": "--", "ms": marker_size}),
    ("no_adaptation", {"label": "No adaptation", "color": "black", "ls": "--", "marker": "*","ms": marker_size}),
])


def load_results_paths():
    root_results_dir = Path("../experiments.ok")
    results = []
    for dataset in root_results_dir.glob("*"):
        for prompt in dataset.glob("*"):
            for model in prompt.glob("*"):
                for base_method in model.glob("*"):
                    for cal_method in base_method.glob("*"):
                        if cal_method.name == ".cache":
                            continue
                        for path in (cal_method / "predictions").glob("*"):
                            if not path.exists() or "test" not in path.name:
                                continue
                            dataset_name, size, seed = dataset.name.split("_")
                            n_shots = load_yaml(f"../configs/prompt/{prompt.name}.yaml")["num_shots"]
                            data_config = load_yaml(f"../configs/dataset/{dataset.name}.yaml")
                            num_samples = data_config["total_train_samples"]
                            model_name = model.name
                            base_method_name = load_yaml(f"../configs/base_method/{base_method.name}.yaml")["method"]
                            cal_method_name = load_yaml(f"../configs/calibration_method/{cal_method.name}.yaml")["method"]
                            split_name = path.name
                            results_path = str(path)
                            results.append({
                                "dataset": dataset_name,
                                "size": int(size),
                                "num_samples": num_samples,
                                "prompt": prompt.name,
                                "n_shots": n_shots,
                                "model": model_name,
                                "base_method": base_method_name,
                                "cal_method": cal_method_name,
                                "split": split_name,
                                "seed": int(seed),
                                "results": results_path
                            })
    df = pd.DataFrame(results)
    df = df.loc[(df["base_method"] != "lora_xval") & (df["cal_method"] != "affine_scalar_no_es"), :].reset_index(drop=True)
    return df

def accuracy(logits, targets):
    return (targets == logits.argmax(axis=1)).mean()

def error_rate(logits, targets):
    return (targets != logits.argmax(axis=1)).mean()

def f1_score(logits, targets):
    return _f1_score(targets, logits.argmax(axis=1), average="macro")

def cross_entropy(logits, targets):
    logprobs = log_softmax(logits, axis=1)
    scores = -logprobs[np.arange(len(targets)), targets]
    scores[scores == np.inf] = 0
    return scores.mean()

def ece(logits, targets):
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = softmax(logits, axis=1)
    confidences = softmaxes.max(axis=1)
    predictions = softmaxes.argmax(axis=1)
    accuracies = predictions == targets

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def _compute_metric(logits, targets, metric, bootstrap_idx=None):
    if metric == "accuracy":
        return accuracy(logits, targets)
    elif metric == "error_rate":
        return error_rate(logits, targets)
    elif metric == "f1_score":
        return f1_score(logits, targets)
    elif metric == "cross_entropy":
        return cross_entropy(logits, targets)
    elif metric == "ece":
        return ece(logits, targets)
    elif metric == "cal_loss_bias":
        logits = torch.from_numpy(logits).float()
        targets = torch.from_numpy(targets).long()
        # return cal_loss(logits, targets, relative=True, condition_ids=bootstrap_idx, learning_rate=0.01, epochs=50, alpha="scalar", beta=True) * 100
        return cal_loss(logits, targets, relative=True, alpha="scalar", beta=True, learning_rate=0.01, epochs=50) * 100
    elif metric == "cal_loss_nobias":
        logits = torch.from_numpy(logits).float()
        targets = torch.from_numpy(targets).long()
        # return cal_loss(logits, targets, relative=True, condition_ids=bootstrap_idx, learning_rate=0.01, epochs=50, alpha="scalar", beta=False) * 100
        return cal_loss(logits, targets, relative=True, alpha="scalar", beta=False, learning_rate=0.01, epochs=50) * 100
    elif metric == "min_calibration_bias":
        logits = torch.from_numpy(logits).float()
        targets = torch.from_numpy(targets).long()
        return min_cal(logits, targets, condition_ids=bootstrap_idx, alpha="scalar", beta=True) * 100
    elif metric == "min_calibration_nobias":
        logits = torch.from_numpy(logits).float()
        targets = torch.from_numpy(targets).long()
        return min_cal(logits, targets, condition_ids=bootstrap_idx, alpha="scalar", beta=False) * 100
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
        logpriors = np.log(priors, where=priors > 0)
        logpriors[priors == 0] = -np.inf
        
        if bootstrap == 0:
            return [
                _compute_metric(logits, targets, metric, bootstrap_idx=None) /
                _compute_metric(logpriors, targets, metric, bootstrap_idx=None)
            ]
        
        rs = np.random.RandomState(random_state)
        values = []
        for bidx in range(bootstrap):
            idx = rs.choice(len(targets), len(targets), replace=True, bootstrap_idx=bidx)
            values.append(
                _compute_metric(logits[idx], targets[idx], metric, bootstrap_idx=bidx) /
                _compute_metric(logpriors[idx], targets[idx], metric, bootstrap_idx=bidx)
            )
        return values
    
    if bootstrap == 0:
        return [_compute_metric(logits, targets, metric, bootstrap_idx=None)]
    
    rs = np.random.RandomState(random_state)
    values = []
    for bidx in range(bootstrap):
        idx = rs.choice(len(targets), len(targets), replace=True, bootstrap_idx=bidx)
        values.append(_compute_metric(logits[idx], targets[idx], metric, bootstrap_idx=bidx))
        
    return values


def compute_results(metrics, bootstrap, random_state):
    df_results = load_results_paths()
    # df_results = df_results[df_results["dataset"] == "sst2"]
    # df_results = df_results[df_results["seed"].isin([639,738,1738,493])]

    grouped = df_results.groupby([c for c in df_results.columns if c not in ['results', 'seed']])
    for group_name, group_df in tqdm(grouped):
        for metric in metrics:
            if f"{metric}:values" not in df_results.columns:
                df_results[f"{metric}:values"] = None
            all_values = []
            for idx, row in group_df.iterrows():
                row_results = load_from_disk(row["results"]).with_format("numpy")
                logits = row_results["logits"].astype(float)
                targets = row_results["label"].astype(int)
                if np.bincount(targets,minlength=logits.shape[1]).min() == 0 and row["split"] == "test":
                    raise ValueError(
                        f"Class missing in {row['dataset']} "
                        f"{row['size']} "
                        f"{row['prompt']} "
                        f"{row['model']} "
                        f"{row['base_method']} "
                        f"{row['cal_method']} "
                        f"{row['split']} "
                    )
                values = compute_metric(logits, targets, metric, bootstrap, random_state)
                all_values.extend(values)
            
            mean = np.mean(all_values)
            median = np.median(all_values)
            std = np.std(all_values)
            q1 = np.quantile(all_values, 0.25)
            q3 = np.quantile(all_values, 0.75)

            for idx, row in group_df.iterrows():
                df_results.loc[idx, metric] = f"{mean:.3f} ± {std:.3f}"
                df_results.loc[idx, f"{metric}:mean"] = mean
                df_results.loc[idx, f"{metric}:median"] = median
                df_results.loc[idx, f"{metric}:std"] = std
                df_results.at[idx, f"{metric}:values"] = np.array(all_values)
                df_results.loc[idx, f"{metric}:Q1"] = q1
                df_results.loc[idx, f"{metric}:Q3"] = q3
    
    df_results.drop(columns=["results", "seed"], inplace=True)
    df_results.drop_duplicates(inplace=True, ignore_index=True, subset=[c for c in df_results.columns if not c.endswith(":values")])

    return df_results

def format_method(base_method, cal_method, dataset, prompt, n_shots):
    method = ""
    kwargs = {"alpha": 1.}
    
    if n_shots == 0:
        method += ""
        kwargs["ls"] = "-"
    else:
        method += "Few-shot "
        kwargs["ls"] = "--"
    
    if prompt == f"basic_{dataset}_{n_shots}-shot_litgpt":
        method += ""
    elif prompt == f"instr_{dataset}_{n_shots}-shot_litgpt":
        method += "(Instructions) "
    elif prompt == f"qa_{dataset}_{n_shots}-shot_litgpt":
        method += "(Question Answering) "
    else:
        raise ValueError(f"Prompt {prompt} is not recognized")
    
    if base_method == "no_adaptation" and cal_method == "no_calibration":
        method += supported_methods["no_adaptation"]["label"]
        kwargs.update(**supported_methods["no_adaptation"])
        kwargs.pop("label")
    elif base_method == "no_adaptation" and cal_method == "affine_scalar":
        method += supported_methods["affine_scalar"]["label"]
        kwargs.update(**supported_methods["affine_scalar"])
        kwargs.pop("label")
    elif base_method == "no_adaptation" and cal_method == "temp_scaling":
        method += supported_methods["temp_scaling"]["label"]
        kwargs.update(**supported_methods["temp_scaling"])
        kwargs.pop("label")
    elif base_method == "no_adaptation" and cal_method == "bias_only":
        method += supported_methods["bias_only"]["label"]
        kwargs.update(**supported_methods["bias_only"])
        kwargs.pop("label")
    elif base_method == "lora" and cal_method == "no_calibration":
        method += supported_methods["lora"]["label"]
        kwargs.update(**supported_methods["lora"])
        kwargs.pop("label")
    elif base_method == "lora_xval" and cal_method == "no_calibration":
        method += supported_methods["lora+no_calibration"]["label"]
        kwargs["color"] = supported_methods["lora+no_calibration"]["color"]
        kwargs["ls"] = supported_methods["lora+no_calibration"]["ls"]
    elif base_method == "lora_xval" and cal_method == "affine_scalar":
        method += supported_methods["lora+affine_scalar"]["label"]
        kwargs["color"] = supported_methods["lora+affine_scalar"]["color"]
        kwargs["ls"] = supported_methods["lora+affine_scalar"]["ls"]
    elif base_method == "lora_xval" and cal_method == "temp_scaling":
        method += supported_methods["lora+temp_scaling"]["label"]
        kwargs["color"] = supported_methods["lora+temp_scaling"]["color"]
        kwargs["ls"] = supported_methods["lora+temp_scaling"]["ls"]
    elif base_method == "lora_xval" and cal_method == "bias_only":
        method += supported_methods["lora+bias_only"]["label"]
        kwargs["color"] = supported_methods["lora+bias_only"]["color"]
        kwargs["ls"] = supported_methods["lora+bias_only"]["ls"]
    elif base_method == "no_adaptation" and cal_method == "affine_scalar_no_es":
        method += supported_methods["affine_scalar_no_es"]["label"]
        kwargs["color"] = supported_methods["affine_scalar_no_es"]["color"]
        kwargs["ls"] = supported_methods["affine_scalar_no_es"]["ls"]
    elif base_method == "lora" and cal_method == "affine_scalar_train_on_val":
        method += supported_methods["lora+affine_scalar_train_on_val"]["label"]
        kwargs.update(**supported_methods["lora+affine_scalar_train_on_val"])
        kwargs.pop("label")
    else:
        raise ValueError(f"Method {base_method} + {cal_method} is not recognized")
    
    return pd.Series({"method": method, "kwargs": kwargs})
    
    

def plot_mean_std_for_model(df, model, metrics, width=.8, err=True, stat="mean"):
    df = df.copy()
    df = df[df["model"] == model]
    df.loc[:,["method", "kwargs"]] = df.apply(lambda x: format_method(x["base_method"], x["cal_method"], x["dataset"], x["prompt"], x["n_shots"]), axis=1)
    methods = [supported_methods[method]["label"] for method in supported_methods if supported_methods[method]["label"] in df["method"].unique()]
    datasets = [dataset for dataset in dataset_short2name.keys() if dataset in df["dataset"].unique()]

    fig, ax = plt.subplots(len(metrics),len(datasets), figsize=(len(datasets)*3, len(metrics)*2), sharex="col")
    if len(metrics) == 1 and len(datasets) == 1:
        ax = np.array([[ax]])
    elif len(metrics) == 1:
        ax = ax[np.newaxis, :]
    elif len(datasets) == 1:
        ax = ax[:, np.newaxis]
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            for m, method in enumerate(methods):
                num_samples = []
                means = []
                stds = []
                sizes = sorted(df.loc[(df["dataset"] == dataset) & (df["method"] == method), "size"].unique())
                for s, size in enumerate(sizes):
                    mask = \
                        (df["dataset"] == dataset) & \
                        (df["size"] == size) & \
                        (df["method"] == method)
                    if mask.sum() == 0:
                        continue
                    if mask.sum() > 1:
                        print(df[mask])
                        raise ValueError("More than one row found")
                    # mean = df[mask][f"{metric}:mean"].values[0]
                    mean = df[mask][f"{metric}:{stat}"].values[0]
                    std = df[mask][f"{metric}:std"].values[0]
                    # num_samples.append(s - width / 2 + width / (len(methods) - 1) * m)
                    num_samples.append(df[mask]["num_samples"].astype(int).values[0])
                    means.append(mean)
                    stds.append(std)
                    kwargs = df.loc[mask, "kwargs"].iloc[0]
                yerr = np.array(stds) if err else None
                ax[j, i].errorbar(
                    np.array(num_samples),
                    np.array(means), 
                    yerr=yerr, 
                    label = method,
                    capsize = width / (len(methods) - 1) * 20,
                    capthick = 2,
                    elinewidth = 2,
                    **kwargs
                )
            ax[j, i].yaxis.grid(True)
            ax[j, i].set_xscale("log")
            ax[j, i].set_xticks([])
            ax[j, i].minorticks_off()
            ylim = ax[j, i].get_ylim()
            sup_lim = 1.2 * df[(df["dataset"] == dataset) & (df["method"] == "No adaptation")].iloc[0][f"{metric}:mean"]
            ax[j, i].set_ylim(max(0,ylim[0]), min(sup_lim, ylim[1]))
            ax[j, i].set_ylim(max(0,ylim[0]), ylim[1])
            ax[j, i].set_yticks(ax[j, i].get_yticks())
            ax[j, i].set_yticklabels([f"{y:2g}" for y in ax[j, i].get_yticks()], fontsize=10)

    for i, dataset in enumerate(datasets):
        sizes = sorted(df.loc[(df["dataset"] == dataset), "size"].unique())
        num_samples = [df.loc[(df["dataset"] == dataset) & (df["size"] == size),"num_samples"].iloc[0] for size in sizes]
        ax[0, i].set_title(
            f"{dataset_short2name[dataset]['name']}\n"
            f"({dataset_short2name[dataset]['num_classes']} classes)",
            fontsize=16
        )
        ax[-1, i].set_xlim(0.8 * min(num_samples), 1.2 * max(num_samples))
        ax[-1, i].set_xticks(num_samples)
        ax[-1, i].set_xticklabels(sizes, fontsize=12)
        ax[-1, i].set_xlabel(f"(x{dataset_short2name[dataset]['num_classes']})", fontsize=14)

    for j, metric in enumerate(metrics):
        if "norm_" in metric:
            metric_name = "N" + metrics_short2name[metric[5:]]
        else:
            metric_name = metrics_short2name[metric]
        ax[j, 0].set_ylabel(metric_name,fontsize=16)

    fig.text(0.5, -0.02, 'Number of adaptation samples', ha='center', fontsize=16)


    hand, lab = ax[0,0].get_legend_handles_labels()
    fig.legend(hand, lab, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(methods), fancybox=True, shadow=True, fontsize=13)
    fig.tight_layout()
    plt.savefig(f"../results_{model}.pdf", dpi=300, bbox_inches="tight")



def plot_manual_boxplot_for_model(df, model, metrics, width=.8):
    df = df.copy()
    df = df[df["model"] == model]
    df.loc[:,["method", "kwargs"]] = df.apply(lambda x: format_method(x["base_method"], x["cal_method"], x["dataset"], x["prompt"], x["n_shots"]), axis=1)
    methods = [supported_methods[method]["label"] for method in supported_methods if supported_methods[method]["label"] in df["method"].unique()]
    datasets = [dataset for dataset in dataset_short2name.keys() if dataset in df["dataset"].unique()]

    fig, ax = plt.subplots(len(metrics),len(datasets), figsize=(len(datasets)*3, len(metrics)*2), sharex="col")
    if len(metrics) == 1 and len(datasets) == 1:
        ax = np.array([[ax]])
    elif len(metrics) == 1:
        ax = ax[np.newaxis, :]
    elif len(datasets) == 1:
        ax = ax[:, np.newaxis]
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            for m, method in enumerate(methods):
                num_samples = []
                medians = []
                stds = []
                yerrs = []
                positions = []
                sizes = sorted(df.loc[(df["dataset"] == dataset) & (df["method"] == method), "size"].unique())
                for size in sizes:
                    mask = \
                        (df["dataset"] == dataset) & \
                        (df["size"] == size) & \
                        (df["method"] == method)
                    if mask.sum() == 0:
                        continue
                    if mask.sum() > 1:
                        print(df[mask])
                        raise ValueError("More than one row found")
                    # mean = df[mask][f"{metric}:mean"].values[0]
                    median = df[mask][f"{metric}:median"].values[0]
                    std = df[mask][f"{metric}:std"].values[0]
                    q1 = df[mask][f"{metric}:Q1"].values[0]
                    q3 = df[mask][f"{metric}:Q3"].values[0]
                    # num_samples.append(s - width / 2 + width / (len(methods) - 1) * m)
                    # num_samples.append(df[mask]["num_samples"].astype(int).values[0])
                    s = np.log2(size)
                    # positions.append(s - width / 2 + width / (len(methods) - 1) * m)
                    positions.append(s)
                    medians.append(median)
                    stds.append(std)
                    yerrs.append([max(0,median - q1), max(0,q3 - median)])
                    kwargs = df.loc[mask, "kwargs"].iloc[0]
                    kwargs["marker"] = "o"
                    kwargs["ms"] = 3
                    # kwargs["alpha"] = 0.7
                    # kwargs["ls"] = "-"
                yerrs = np.array(yerrs).T
                ax[j, i].errorbar(
                    np.array(positions),
                    np.array(medians),
                    yerr=yerrs,
                    label = method,
                    capsize = width / (len(methods) - 1) * 20,
                    # capthick = 2,
                    # elinewidth = 2,
                    **kwargs
                )
            ax[j, i].yaxis.grid(True)
            ax[j, i].set_xscale("log")
            ax[j, i].set_xticks([])
            ax[j, i].minorticks_off()
            ylim = ax[j, i].get_ylim()
            sup_lim = 1.2 * df[(df["dataset"] == dataset) & (df["method"] == "No adaptation")].iloc[0][f"{metric}:mean"]
            # ax[j, i].set_ylim(ylim[0], min(sup_lim, ylim[1]))
            ax[j, i].set_ylim(max(0,ylim[0]), min(sup_lim, ylim[1]))
            ax[j, i].set_yticks(ax[j, i].get_yticks())
            ax[j, i].set_yticklabels([f"{y:2g}" for y in ax[j, i].get_yticks()], fontsize=10)

    for i, dataset in enumerate(datasets):
        sizes = sorted(df.loc[(df["dataset"] == dataset), "size"].unique())
        # num_samples = [df.loc[(df["dataset"] == dataset) & (df["size"] == size),"num_samples"].iloc[0] for size in sizes]
        ax[0, i].set_title(
            f"{dataset_short2name[dataset]['name']}\n"
            f"({dataset_short2name[dataset]['num_classes']} classes)",
            fontsize=16
        )
        # ax[-1, i].set_xlim(0.8 * min(num_samples), 1.2 * max(num_samples))
        # ax[-1, i].set_xticks(num_samples)
        positions = [np.log2(size) for size in sizes]
        ax[-1, i].set_xlim(0.95 * min(positions), 1.05 * max(positions))
        ax[-1, i].set_xticks(positions)
        ax[-1, i].set_xticklabels(sizes, fontsize=12)
        ax[-1, i].set_xlabel(f"(x{dataset_short2name[dataset]['num_classes']})", fontsize=14)

    for j, metric in enumerate(metrics):
        if "norm_" in metric:
            metric_name = "N" + metrics_short2name[metric[5:]]
        else:
            metric_name = metrics_short2name[metric]
        ax[j, 0].set_ylabel(metric_name, fontsize=16)

    fig.text(0.5, -0.02, 'Number of adaptation samples', ha='center', fontsize=16)


    hand, lab = ax[0,0].get_legend_handles_labels()
    fig.legend(hand, lab, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(methods), fancybox=True, shadow=True, fontsize=13)
    fig.tight_layout()
    plt.savefig(f"../results_{model}_intervals.pdf", dpi=300, bbox_inches="tight")


def boxplot_for_model(df, model, metrics, width=.8):
    df = df.copy()
    df = df[df["model"] == model]
    df.loc[:,["method", "kwargs"]] = df.apply(lambda x: format_method(x["base_method"], x["cal_method"], x["dataset"], x["prompt"], x["n_shots"]), axis=1)
    methods = [supported_methods[method]["label"] for method in supported_methods if supported_methods[method]["label"] in df["method"].unique()]
    datasets = [dataset for dataset in dataset_short2name.keys() if dataset in df["dataset"].unique()]

    fig, ax = plt.subplots(len(metrics),len(datasets), figsize=(len(datasets)*3, len(metrics)*2), sharex="col")
    if len(metrics) == 1 and len(datasets) == 1:
        ax = np.array([[ax]])
    elif len(metrics) == 1:
        ax = ax[np.newaxis, :]
    elif len(datasets) == 1:
        ax = ax[:, np.newaxis]
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            for m, method in enumerate(methods):
                positions = []
                all_values = []
                sizes = sorted(df.loc[(df["dataset"] == dataset) & (df["method"] == method), "size"].unique())
                for size in sizes:
                    mask = \
                        (df["dataset"] == dataset) & \
                        (df["size"] == size) & \
                        (df["method"] == method)
                    if mask.sum() == 0:
                        continue
                    if mask.sum() > 1:
                        print(df[mask])
                        raise ValueError("More than one row found")
                    all_values.append(df[mask][f"{metric}:values"].values[0])
                    s = np.log2(size)
                    positions.append(s - width / 2 + width / (len(methods) - 1) * m)
                    # num_samples.append(df[mask]["num_samples"].astype(int).values[0])
                    kwargs = df.loc[mask, "kwargs"].iloc[0]
                all_values = np.vstack(all_values).T
                bplot = ax[j, i].boxplot(
                    x=all_values,
                    positions=positions,
                    widths=width / (len(methods) - 1),
                    showfliers=False,
                    patch_artist=True,
                    # label = method,
                    # capsize = width / (len(methods) - 1) * 20,
                    # capthick = 2,
                    # elinewidth = 2,
                    # **kwargs
                )
                for patch in bplot['boxes']:
                    patch.set_facecolor(kwargs["color"])
            ax[j, i].yaxis.grid(True)
            ax[j, i].set_xscale("log")
            ax[j, i].set_xticks([])
            ax[j, i].minorticks_off()
            # ylim = ax[j, i].get_ylim()
            # sup_lim = 1.2 * df[(df["dataset"] == dataset) & (df["method"] == "No adaptation")].iloc[0][f"{metric}:mean"]
            # ax[j, i].set_ylim(ylim[0], min(sup_lim, ylim[1]))

    for i, dataset in enumerate(datasets):
        sizes = sorted(df.loc[(df["dataset"] == dataset), "size"].unique())
        xticks = [np.log2(size) for size in sizes]
        # num_samples = [df.loc[(df["dataset"] == dataset) & (df["size"] == size),"num_samples"].iloc[0] for size in sizes]
        ax[0, i].set_title(
            f"{dataset_short2name[dataset]['name']}\n"
            f"({dataset_short2name[dataset]['num_classes']} classes)"
        )
        # ax[-1, i].set_xlim(0.8 * min(num_samples), 1.2 * max(num_samples))
        # ax[-1, i].set_xticks(num_samples)
        ax[-1, i].set_xlim(0.8 * min(xticks), 1.2 * max(xticks))
        ax[-1, i].set_xticks(xticks)
        ax[-1, i].set_xticklabels(sizes)
        ax[-1, i].set_xlabel(f"(x{dataset_short2name[dataset]['num_classes']})")

    for j, metric in enumerate(metrics):
        if "norm_" in metric:
            metric_name = "N" + metrics_short2name[metric[5:]]
        else:
            metric_name = metrics_short2name[metric]
        ax[j, 0].set_ylabel(metric_name)

    fig.text(0.5, 0.0, 'Number of training samples', ha='center')


    hand, lab = ax[0,0].get_legend_handles_labels()
    fig.legend(hand, lab, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(methods), fancybox=True, shadow=True)
    fig.tight_layout()