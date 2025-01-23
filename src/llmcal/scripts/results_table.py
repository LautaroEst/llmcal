
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ..src.utils import load_yaml
from .results_vs_samples import DATASETS, metric2name, process_data

highlights = {
    "nce": lambda x: f"$\\mathbf{{{x}}}$",
    "ner": lambda x: f"$\\underline{{{x}}}$",
}

def create_table(results_dir, methods, metrics, methods_config, datasets, sizes):

    data_all = pd.DataFrame([
        {"dataset": dataset, "size": size, "method": method, "result": ""}
        for size in [sizes[0], sizes[-1]]
        for method in methods
        for dataset in datasets
    ]).pivot(index=["size","method"], columns="dataset", values="result")
    index = [(size, method) for size in [sizes[0],sizes[-1]] for method in methods]
    data_all = data_all.reindex(index)

    for metric in metrics:
        data = pd.read_json(results_dir / f"{metric}.jsonl", orient='records', lines=True)
        data = process_data(data, datasets, sizes, methods)
        no_adaptation = data[data["method"] == "no_adaptation"]
        min_size_no_adaptation = no_adaptation.copy()
        min_size_no_adaptation["size"] = sizes[0]
        max_size_no_adaptation = no_adaptation.copy()
        max_size_no_adaptation["size"] = sizes[-1]
        data = data[data["method"] != "no_adaptation"]
        data = pd.concat([data, min_size_no_adaptation, max_size_no_adaptation], ignore_index=True)
        data = data[data["size"].isin([sizes[0],sizes[-1]])].reset_index(drop=True)

        data["result"] = data["median"].apply(lambda x: f"{x:.2f}")
        best_idx = data.groupby(["dataset","size"])["median"].idxmin().values
        data.loc[best_idx,"result"] = data.loc[best_idx,"result"].apply(highlights[metric])
        data = data.pivot(index=["size","method"], columns="dataset", values="result")
        index = [(size, method) for size in [sizes[0],sizes[-1]] for method in methods]
        data = data.reindex(index)
        data = data.fillna("N/A")

        for size in [sizes[0],sizes[-1]]:
            for method in methods:
                for dataset in datasets:
                    if data_all.loc[(size, method), dataset] == "":
                        data_all.loc[(size, method), dataset] += data.loc[(size, method), dataset]
                    else:
                        data_all.loc[(size, method), dataset] += " / " + data.loc[(size, method), dataset]
    
    data_all.index = data.index.map(lambda x: ({sizes[0]: f"min (N = {int(np.log2(sizes[0]))})", sizes[-1]: f"max (N = {int(np.log2(sizes[-1]))})"}[x[0]], methods_config[x[1]]["label"].replace("%","\\%").replace("\n", " &" * len(datasets) + " \\\\\n & ") ))
    data_all = data_all.loc[:,datasets]
    data_all.columns = data_all.columns.map(lambda x: DATASETS[x]["name"])
    data_all.columns.name = None
    data_all.index.names = [None, None]

    return data_all
    
        

def main(
    datasets,
    sizes,
    metrics,
    methods,
    methods_config,
    results_dir,
    output_path
):
    datasets = list(map(str, datasets.split()))
    sizes = list(map(int, sizes.split()))
    methods = list(map(str, methods.split()))
    methods_config = load_yaml(methods_config)
    metrics = list(map(str, metrics.split()))
    output_path = Path(output_path)
    output_dir = output_path.parent
    results_dir = Path(results_dir)
    
    
    table = create_table(results_dir, methods, metrics, methods_config, datasets, sizes)
    table.to_latex(output_path, escape=False)

    
    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)