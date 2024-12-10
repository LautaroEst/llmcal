
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def process_data(data, datasets, sizes, methods):

    # Keep matched trainings
    data = data[data["train_dataset"] == data["test_dataset"]]

    # Keep evaluation in test
    data = data[data["test_lst"].str.startswith("test_")]

    # Keep only selected datasets
    data = data[data["test_dataset"].isin(datasets)]

    # Keep only selected sizes
    data = data[data["size"].isin(sizes)]

    # Keep only selected methods
    data = data[data["method"].isin(methods)]

    # Replace method name for full description
    data["full_method"] = data.apply(lambda x: f"{x['method']} (train={x['train_lst']}, val={x['val_lst']})", axis=1)

    # Clean dataframe
    data = data.drop(columns=["train_dataset", "method", "train_lst", "val_lst", "test_lst"])
    data = data.rename(columns={"full_method": "method"})
    data = data.rename(columns={"test_dataset": "dataset"})
    data = data.loc[:, ["dataset", "method", "size", "seed", "result", "min_result"]]
    
    # Group by size
    data = data.groupby(["dataset", "method", "size"])["result"].agg(
        median=lambda x: x.median(),
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ).reset_index()

    return data
    




def plot_metric_vs_samples(data, metric, datasets, sizes, output_path):
    
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
            ax[i].plot(method_data.index, method_data["median"], label=method, color=f"C{j}", marker="o")
            ax[i].fill_between(method_data.index, method_data["q1"], method_data["q3"], alpha=0.3, color=f"C{j}")
        
        ax[i].set_title(dataset)
        ax[i].set_xlabel("Samples")
        ax[i].set_xscale("log")
        ax[i].set_xticks(sizes)
        ax[i].set_xticklabels(sizes)
        ax[i].grid(axis="y")
        ax[i].set_xlim([min(sizes), max(sizes)])
    
    ax[0].set_ylabel(f"{metric}")
    ax[0].legend(loc="upper left", bbox_to_anchor=(-1, 1))
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    


def main(
    datasets,
    sizes,
    metric,
    methods,
    results_path,
    output_dir,
):
    datasets = list(map(str, datasets.split()))
    sizes = list(map(int, sizes.split()))
    methods = list(map(str, methods.split()))
    output_dir = Path(output_dir)

    data = pd.read_json(results_path, orient='records', lines=True)
    data = process_data(data, datasets, sizes, methods)
    data.to_csv(output_dir / f"{metric}.csv", index=False)
    plot_metric_vs_samples(data, metric, datasets, sizes, output_dir / f"{metric}.png")

    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)