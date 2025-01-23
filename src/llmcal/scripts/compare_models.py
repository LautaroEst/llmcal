
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ..src.utils import load_yaml
from .results_vs_samples import DATASETS, metric2name, process_data, plot_metric_vs_samples

method2style = {
    "no_adaptation": "-",
    "lora_1.0_no_es_plus_tempscaling": "--",
    "dp_calibration": "-.",
}

def main(
    datasets,
    metrics,
    sizes,
    methods_config,
    output_path,
    models,
    results_dirs,
    intervals,
    methods,
):
    datasets = list(map(str, datasets.split()))
    sizes = list(map(int, sizes.split()))
    models = list(map(str, models.split()))
    methods = list(map(str, methods.split()))
    methods_config = load_yaml(methods_config)
    metrics = list(map(str, metrics.split()))
    output_path = Path(output_path)
    output_dir = output_path.parent
    results_dirs = list(map(Path, results_dirs.split()))

    fig, axs = plt.subplots(len(metrics), len(datasets), figsize=(6 * len(datasets), 12))
    processed_data = {}
    custom_handles = []
    all_datasets_data = {}
    for i, (model, results_dir) in enumerate(zip(models, results_dirs)):
        for method in methods:
            methods_config[method]["color"] = f"C{i}"
            methods_config[method]["linestyle"] = method2style[method]
        for ax, metric in zip(axs,metrics):
            data = pd.read_json(results_dir / f"{metric}.jsonl", orient='records', lines=True)
            processed_data[metric] = data
            data = process_data(data, datasets, sizes, methods)
            datasets_data = plot_metric_vs_samples(ax, data, methods, methods_config, datasets, sizes, intervals=intervals, pos=i/10)
            all_datasets_data[(model,metric)] = datasets_data
            data.to_csv(output_dir / f"{metric}.csv", index=False)
            ax[0].set_ylabel(f"{metric2name[metric]}", fontsize=22)

    for j, dataset in enumerate(datasets):
        axs[0,j].set_title(DATASETS[dataset]["name"], fontsize=22)
    
    fig.text(0.5, 0.04, 'Number of train samples', ha='center', fontsize=22)

    # Gather handles and labels from all axes
    custom_handles = []
    for i, model in enumerate(models):
        custom_handles.append(
            plt.Line2D([0], [0], color=f"C{i}", linestyle="-", marker="o", markersize=5, label=model)
        ) 
    for method in methods:
        custom_handles.append(
            plt.Line2D([0], [0], color="black", linestyle=method2style[method], label=methods_config[method]["label"])
        )
    fig.legend(handles=custom_handles, loc='upper right', bbox_to_anchor=(1.08, .95), title_fontsize=24, fontsize=22)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    from fire import Fire
    Fire(main)