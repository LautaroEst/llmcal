
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..src.utils import load_yaml


DATASETS = {
    "sst2": {"name":  "SST-2", "num_classes": 2},
    "agnews": {"name":  "AGNews", "num_classes": 4},
    "dbpedia": {"name":  "DBPedia", "num_classes": 14},
    "20newsgroups": {"name":  "20 Newsgroups", "num_classes": 20},
    "banking77": {"name":  "Banking77", "num_classes": 77},
}

metric2name = {
    "nce": "NCE",
    "ner": "NER",
}

def read_data(results_dir: Path, metrics: List[str]):
    dfs = []
    for metric in metrics:
        df = pd.read_json(results_dir / f"{metric}.jsonl", orient='records', lines=True)
        df = df.rename(columns={"result": f"{metric}"})
        df = df.drop(columns=["min_result"])
        dfs.append(df)
    
    # merge dataframes with columns [dataset, method, size, seed, nce] and [dataset, method, size, seed, ner] on all columns but last
    data = dfs[0]
    for df in dfs[1:]:
        data = data.merge(df, on=["dataset", "method", "size", "seed"], how="outer")
    return data

def plot_bars(data, methods_config, output_path, datasets, metrics, methods, sizes):
    fig, ax = plt.subplots(len(metrics), len(datasets), figsize=(6 * len(datasets), 12), sharex=True)

    data_adapted = data.loc[data["method"] != "no_adaptation"]
    adapted_methods = [m for m in methods if m != "no_adaptation"]
    data_no_adapt = data.loc[data["method"] == "no_adaptation"]


    n_methods = len(methods) + 1
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            # plot bar groups. One group per size. Each bar is a method
            medians = []
            for k, method in enumerate(adapted_methods):
                for s, size in enumerate(sizes):
                    method_data = data_adapted.loc[data_adapted["method"] == method]
                    method_data = method_data.loc[method_data["dataset"] == dataset]
                    method_data = method_data.loc[method_data["size"] == size]
                    median = method_data.groupby("size")[metric].median().values[0]
                    medians.append(median.max())
                    q1 = method_data.groupby("size")[metric].quantile(0.25).values[0]
                    q3 = method_data.groupby("size")[metric].quantile(0.75).values[0]
                    # alpha = 0.5 if "SFT+PHC" in methods_config[method]["label"] else 1.0
                    alpha = 0.7 if "SFT " in methods_config[method]["label"] else 1.0
                    if "SFT " in methods_config[method]["label"]:
                        hatch = "x"
                    elif "SFT+PHC" in methods_config[method]["label"]:
                        hatch = "/"
                    else:
                        hatch = None
                    ax[i, j].bar(s + k / n_methods, median, yerr=[[median - q1], [q3 - median]], label=methods_config[method]["label"], width=0.8 / n_methods, color=methods_config[method]["color"], alpha=alpha, hatch=hatch)
                    ax[i,j].set_xticks(range(len(sizes)))
                    ax[i,j].set_xticklabels([f"{' '*20}T = {int(np.log2(size))}" for size in sizes])
            # plot no adaptation
            no_adapt_value = data_no_adapt.loc[data_no_adapt["dataset"] == dataset,metric].values[0]
            xlims = ax[i,j].get_xlim()
            ax[i,j].plot(xlims, [no_adapt_value] * len(sizes), label=methods_config["no_adaptation"]["label"], color=methods_config["no_adaptation"]["color"], linestyle="--")
            ax[i,j].set_xticks(ax[i,j].get_xticks())
            ax[i,j].set_xticklabels(ax[i,j].get_xticklabels(), fontsize=20)
            ax[i,j].set_xlim(xlims)
            ax[i,j].set_ylim(0, min(1, 1.05 * max(medians)))
            ax[i,j].set_yticks(ax[i,j].get_yticks())
            ax[i,j].set_yticklabels(ax[i,j].get_yticklabels(), fontsize=20)
            ax[i,j].grid(True)
    
    for i, dataset in enumerate(datasets):
        ax[0, i].set_title(DATASETS[dataset]["name"], fontsize=22)
    for j, metric in enumerate(metrics):
        ax[j, 0].set_ylabel(f"{metric2name[metric]}", fontsize=22)

    fig.text(0.5, -0.05, 'Adaptation sizes', ha='center', fontsize=22)

    # Gather handles and labels from all axes
    handles, labels = [], []
    for a in ax.flat:
        hs, ls = a.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, .95), title="Method", title_fontsize=24, fontsize=22)

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)




def main(
    datasets,
    metrics,
    sizes,
    methods_config,
    results_dir,
    output_path,
    methods,
):
    metrics = list(map(str, metrics.split()))
    datasets = list(map(str, datasets.split()))
    methods = list(map(str, methods.split()))
    sizes = list(map(int, sizes.split()))
    sizes = [sizes[0], sizes[-1]]
    results_dir = Path(results_dir)
    methods_config = load_yaml(methods_config)

    data = read_data(results_dir, metrics)
    plot_bars(data, methods_config, output_path, datasets, metrics, methods, sizes)

if __name__ == "__main__":
    from fire import Fire
    Fire(main)