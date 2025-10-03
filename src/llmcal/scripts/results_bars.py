
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..src.utils import load_yaml
from .results_vs_samples import compute_num_samples


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
    "nbs": "NBS",
    "cal_err": "Calibration error",
    "ece": "ECE",
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

def plot_bars(data, methods_config, output_path, datasets, metrics, methods, sizes, no_adaptation="auto", fontsize_noa=22, pos=0):
    fig, ax = plt.subplots(len(metrics), len(datasets), figsize=(6 * len(datasets), 12), sharex=False)

    data_adapted = data.loc[data["method"] != "no_adaptation"]
    adapted_methods = [m for m in methods if m != "no_adaptation"]
    data_no_adapt = data.loc[data["method"] == "no_adaptation"]

    i_ax, j_ax = 0, 0
    n_methods = len(methods) + 1
    medians = {}
    for i, metric in enumerate(metrics):
        medians[metric] = []
        for j, dataset in enumerate(datasets):
            # plot bar groups. One group per size. Each bar is a method
            for k, method in enumerate(adapted_methods):
                for s, size in enumerate(sizes):
                    method_data = data_adapted.loc[data_adapted["method"] == method]
                    method_data = method_data.loc[method_data["dataset"] == dataset]
                    method_data = method_data.loc[method_data["size"] == size]
                    median = method_data.groupby("size")[metric].median().values[0]
                    medians[metric].append(median.max())
                    q1 = method_data.groupby("size")[metric].quantile(0.25).values[0]
                    q3 = method_data.groupby("size")[metric].quantile(0.75).values[0]
                    # alpha = 0.5 if "SFT+PHC" in methods_config[method]["label"] else 1.0
                    # alpha = 0.7 if "SFT " in methods_config[method]["label"] else 1.0
                    # if "SFT " in methods_config[method]["label"]:
                    #     hatch = "x"
                    # elif "SFT+PHC" in methods_config[method]["label"]:
                    #     hatch = "/"
                    # else:
                    #     hatch = None
                    alpha = None
                    hatch = None
                    ax[i, j].bar(s + k / n_methods, median, yerr=[[median - q1], [q3 - median]], label=methods_config[method]["label"], width=0.8 / n_methods, color=methods_config[method]["color"], alpha=alpha, hatch=hatch)
                    # ax[i,j].set_xticks(range(len(sizes)))
                    
    # plot no adaptation
    for i, metric in enumerate(metrics):
        y_max = np.round(min(1.4, 1.05 * max(medians[metric])),1)
        for j, dataset in enumerate(datasets):
            dataset_data = data_no_adapt.loc[data_no_adapt["dataset"] == dataset]
            method_data = dataset_data.loc[dataset_data["method"] == "no_adaptation"]
            # min_q1 = data_adapted[data_adapted["dataset"] == dataset].groupby("size")[metric].quantile(0.25).min()
            # max_median = data_adapted[data_adapted["dataset"] == dataset].groupby("size")[metric].median().max()
            if method_data.loc[:, metric].item() < y_max and no_adaptation in ["plot", "auto"]:
                num_samples = [-1/n_methods, len(sizes) - 1 + len(adapted_methods)/n_methods]
                noa_medians = [method_data.loc[:, metric].item()] * len(num_samples)
                ax[i,j].plot(num_samples, noa_medians, label=methods_config["no_adaptation"]["label"], color=methods_config["no_adaptation"]["color"], linestyle=methods_config["no_adaptation"]["linestyle"])
                i_ax = i
                j_ax = j
                # print(metric, dataset)
            elif no_adaptation in ["text", "auto"]:
                text = f"{methods_config['no_adaptation']['label']}"
                ax[i,j].text(.95, .95-pos, 
                    f"{text} = {method_data.loc[:, metric].item():.2f}",
                    fontsize=fontsize_noa, ha="right", va="top", transform=ax[i,j].transAxes, color=methods_config["no_adaptation"]["color"]
                )


            # no_adapt_value = data_no_adapt.loc[data_no_adapt["dataset"] == dataset,metric].values[0]
            xlims = [-1/n_methods, len(sizes) - 1 + len(adapted_methods)/n_methods]
            # ax[i,j].plot(xlims, [no_adapt_value] * len(sizes), label=methods_config["no_adaptation"]["label"], color=methods_config["no_adaptation"]["color"], linestyle="--")
            ax[i,j].set_xlim(xlims)
            # ax[i,j].set_xticks(range(len(sizes)))
            # ax[i,j].set_xticklabels(ax[i,j].get_xticklabels(), fontsize=26)
            
            ax[i,j].set_ylim(0, y_max)
            if j == 0:
                ax[i,j].set_ylabel(f"{metric2name[metric]}", fontsize=30)
                ax[i,j].set_yticks(np.arange(0,int(y_max*10+1),2)/10)
                ax[i,j].set_yticklabels([f"{d:.1f}" for d in np.arange(0,int(y_max*10+1),2)/10], fontsize=24)
                # ax[i,j].set_yticks(ax[i,j].get_yticks())
                # ax[i,j].set_yticklabels([f"{d:.1f}" for d in ax[i,j].get_yticks()], fontsize=24)
            else:
                # ax[i,j].sharey(ax[i,0])
                ax[i,j].set_yticks([])
                ax[i,j].set_yticklabels([])
            
            
            # ax[i,j].set_yticks(ax[i,j].get_yticks())
            # ax[i,j].set_yticklabels(ax[i,j].get_yticklabels(), fontsize=24)
            # ax[i,j].grid(axis="y")
    
    #YES
    for i, dataset in enumerate(datasets):
        ax[0, i].set_title(DATASETS[dataset]["name"], fontsize=30)
        ax[0, i].set_xticks([])
        ax[-1, i].set_xticks(range(len(sizes)))
        ax[-1, i].set_xticklabels([f"{' '*15}N = {size}" for size in compute_num_samples(sizes, dataset)], fontsize=26)

    
    
    # for j, metric in enumerate(metrics):
    #     ax[j, 0].set_ylabel(f"{metric2name[metric]}", fontsize=30)
    #     ax[j, 0].set_yticks(ax[j, 0].get_yticks())
    #     ax[j, 0].set_yticklabels(ax[j, 0].get_yticklabels(), fontsize=24)

    fig.text(0.5, -0.05, 'Adaptation sizes', ha='center', fontsize=26)

    # Gather handles and labels from all axes
    labels = [methods_config[method]["label"] for method in methods]
    handles = []
    hs, ls = ax[i_ax,j_ax].get_legend_handles_labels()
    for l in labels:
        i = 0
        while ls[i] != l:
            i += 1
        handles.append(hs[i])
    remaining = len(handles) % 4
    for i in range(remaining):
        handles.append(plt.Line2D([], [], color='none', label=''))
        labels.append('')

    

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.72, -0.3), title="Method", ncol=4, title_fontsize=30, fontsize=28)

    fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=-8.0)
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