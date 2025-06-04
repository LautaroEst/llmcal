
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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

        


def process_data(data, datasets, sizes, methods):

    # Keep only selected datasets
    data = data[data["dataset"].isin(datasets)]

    # Keep only selected sizes
    data = data[data["size"].isin(sizes) | (data["size"] == 'all')]

    # Keep only selected methods
    data = data[data["method"].isin(methods)]

    # Group by size
    data = data.groupby(["dataset", "method", "size"])["result"].agg(
        median=lambda x: x.median(),
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        count=lambda x: x.count(),
    ).reset_index()
    
    assert (data["count"] > 0).all()
    data = data.drop(columns=["count"])

    return data
    

def compute_num_samples(sizes, dataset):
    num_classes = DATASETS[dataset]["num_classes"]
    scale = sizes / np.log2(num_classes)
    nearest_power_of_2 = 2 ** np.round(np.log2(scale))
    num_samples = nearest_power_of_2 * num_classes
    return num_samples.astype(int)



def plot_metric_vs_samples(ax, data, all_methods, methods_config, datasets, sizes, intervals=False, pos=0, no_adaptation="plot", modelname_noa=None, fontsize_noa=18):
    datasets_data = {}
    for i, dataset in enumerate(datasets):
        
        # All methods in dataset
        dataset_data = data[data["dataset"] == dataset]
        methods = [m for m in all_methods if m in dataset_data["method"].unique()]

        for j, method in enumerate(methods):
            # Get data for method
            method_data = dataset_data[dataset_data["method"] == method].set_index("size").drop(columns=["dataset", "method"])
            
            # Fill missing sizes
            missing_sizes = set(sizes) - set(method_data.index) if method != "no_adaptation" else set()
            for size in missing_sizes:
                method_data.loc[size] = [np.nan, np.nan, np.nan]

            # Sort by size
            method_data = method_data.sort_index()

            # Plot
            if method == "no_adaptation":
                if dataset_data.loc[:, "q1"].min() < method_data.loc["all", "median"] < dataset_data.loc[:, "median"].max() and no_adaptation in ["plot", "auto"]:
                    num_samples = compute_num_samples(sizes, dataset)
                    medians = [method_data.loc["all", "median"]] * len(num_samples)
                    q1 = [method_data.loc["all", "q1"]] * len(num_samples)
                    q3 = [method_data.loc["all", "q3"]] * len(num_samples)
                    kwargs = methods_config[method]
                    ax[i].plot(num_samples, medians, **kwargs)
                elif no_adaptation in ["text", "auto"]:
                    if modelname_noa is not None:
                        text = f"{methods_config['no_adaptation']['label']} ({modelname_noa})"
                    else:
                        text = f"{methods_config['no_adaptation']['label']}"
                    ax[i].text(.95, .95-pos, 
                        f"{text} = {method_data.loc['all', 'median']:.2f}",
                        fontsize=fontsize_noa, ha="right", va="top", transform=ax[i].transAxes, color=methods_config[method]["color"]
                    )
                elif no_adaptation == "skip":
                    pass

            else:
                num_samples = compute_num_samples(method_data.index.astype(int), dataset)
                medians = method_data["median"]
                q1 = method_data["q1"]
                q3 = method_data["q3"]
                kwargs = methods_config[method]
                ax[i].plot(num_samples, medians, **kwargs)
                if intervals:
                    ax[i].fill_between(num_samples, q1, q3, alpha=0.3, color=kwargs["color"])
        
        
        ax[i].set_xscale("log")
        ax[i].set_xticks(num_samples)
        ax[i].set_xticklabels(num_samples, fontsize=18)
        ax[i].set_xlim([min(num_samples)*0.9, max(num_samples)*1.1])

        

        datasets_data[dataset] = dataset_data

    return datasets_data


def main(
    datasets,
    sizes,
    metrics,
    methods,
    methods_config,
    results_dir,
    output_path,
    intervals = False,
):
    datasets = list(map(str, datasets.split()))
    sizes = list(map(int, sizes.split()))
    methods = list(map(str, methods.split()))
    methods_config = load_yaml(methods_config)
    metrics = list(map(str, metrics.split()))
    output_path = Path(output_path)
    output_dir = output_path.parent
    results_dir = Path(results_dir)

    fig, axs = plt.subplots(len(metrics), len(datasets), figsize=(6 * len(datasets), 12))
    processed_data = {}
    for ax, metric in zip(axs,metrics):
        data = pd.read_json(results_dir / f"{metric}.jsonl", orient='records', lines=True)
        processed_data[metric] = data
        data = process_data(data, datasets, sizes, methods)
        datasets_data = plot_metric_vs_samples(ax, data, methods, methods_config, datasets, sizes, intervals=intervals, no_adaptation="auto")
        for i, dataset in enumerate(datasets):
            min_y, max_y = datasets_data[dataset].loc[datasets_data[dataset]["method"].isin(set(methods) - {"no_adaptation"}),"median"].min(), datasets_data[dataset].loc[datasets_data[dataset]["method"].isin(set(methods) - {"no_adaptation"}),"median"].max()
            ax[i].set_ylim(min_y*0.99, max_y*1.01)
            ax[i].set_yticks(np.round(ax[i].get_yticks(),3))
            ax[i].set_yticklabels(ax[i].get_yticks(), fontsize=16)
            # ax[i].grid(axis="y")

        data.to_csv(output_dir / f"{metric}.csv", index=False)
        ax[0].set_ylabel(f"{metric2name[metric]}", fontsize=22)
    for j, dataset in enumerate(datasets):
        axs[0,j].set_title(DATASETS[dataset]["name"], fontsize=22)
    
    fig.text(0.5, 0.04, 'Number of train samples', ha='center', fontsize=22)
    # axs[0,-1].legend(loc="upper right", bbox_to_anchor=(2.4, 1), title="Method", title_fontsize=24, fontsize=22)

    # Gather handles and labels from all axes
    handles, labels = [], []
    for ax in axs.flat:
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -0.1), ncols=5, title="Method", title_fontsize=28, fontsize=26)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)