
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


        


def process_data(data, datasets, sizes, methods):

    # Keep only selected datasets
    data = data[data["dataset"].isin(datasets)]

    # Keep only selected sizes
    data = data[data["size"].isin(sizes)]

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



def plot_metric_vs_samples(data, metric, methods_config, datasets, sizes, output_path):
    
    fig, ax = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4))
    for i, dataset in enumerate(datasets):
        
        # All methods in dataset
        dataset_data = data[data["dataset"] == dataset]
        methods = sorted(dataset_data["method"].unique())

        for j, method in enumerate(methods):
            # Get data for method
            method_data = dataset_data[dataset_data["method"] == method].set_index("size").drop(columns=["dataset", "method"])
            
            # Fill missing sizes
            missing_sizes = set(sizes) - set(method_data.index)
            for size in missing_sizes:
                method_data.loc[size] = [np.nan, np.nan, np.nan]

            # Sort by size
            method_data = method_data.sort_index()

            # Plot
            num_samples = compute_num_samples(method_data.index, dataset)
            kwargs = methods_config[method]
            ax[i].plot(num_samples, method_data["median"], **kwargs)
            ax[i].fill_between(num_samples, method_data["q1"], method_data["q3"], alpha=0.3, color=kwargs["color"])
        
        ax[i].set_title(DATASETS[dataset]["name"])
        ax[i].set_xlabel("Samples")
        ax[i].set_xscale("log")
        ax[i].set_xticks(num_samples)
        ax[i].set_xticklabels(num_samples)
        ax[i].set_xlim([min(num_samples)*0.9, max(num_samples)*1.1])

        min_y, max_y = dataset_data["median"].min(), min(dataset_data["median"].max(),1.)
        ax[i].set_ylim(min_y*0.9, max_y*1.1)
        ax[i].grid(axis="y")
    
    ax[0].set_ylabel(f"{metric}")
    ax[0].legend(loc="upper left", bbox_to_anchor=(-1, 1))
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(
    datasets,
    sizes,
    metric,
    methods,
    methods_config,
    results_path,
    output_dir,
):
    datasets = list(map(str, datasets.split()))
    sizes = list(map(int, sizes.split()))
    methods = list(map(str, methods.split()))
    methods_config = load_yaml(methods_config)
    output_dir = Path(output_dir)

    data = pd.read_json(results_path, orient='records', lines=True)
    data = process_data(data, datasets, sizes, methods)
    data.to_csv(output_dir / f"{metric}.csv", index=False)
    plot_metric_vs_samples(data, metric, methods_config, datasets, sizes, output_dir / f"{metric}.png")

    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)