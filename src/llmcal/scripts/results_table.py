
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ..src.utils import load_yaml
from .results_vs_samples import DATASETS, metric2name, process_data

def highlight_local(x):
    if x['group'] == "noa":
        return str(x['result'])
    return f"$\\mathbf{{{x['result']}}}$"

def highlight_global(x):
    return f"\\underline{{{x['result']}}}"

def get_all_mins(x):
    import pdb; pdb.set_trace()
    return x[x.round(2) == x.round(2).min()].index.to_numpy()

def method2group(method):
    if method == "no_adaptation":
        return "noa"
    elif method in ["temp_scaling", "dp_calibration", "bias_shift", "vector_scaling"]:
        return "phc"
    elif method.startswith("lora") and "plus" not in method:
        return "sft"
    elif method.startswith("lora") and "plus" in method:
        return "sft+phc"

def create_table(results_dir, methods, metrics, methods_config, datasets, sizes):

    methods = [m for m in methods if m != "no_adaptation"]
    # data_all = pd.DataFrame([
    #     {"dataset": dataset, "size": size, "method": method, "group": method2group(method), "result": ""}
    #     for size in [sizes[0], sizes[-1]]
    #     for method in methods
    #     for dataset in datasets
    # ]).pivot(index=["size","group", "method"], columns="dataset", values="result")
    # index = [(size, method2group(method), method) for size in [sizes[0],sizes[-1]] for method in methods]
    # data_all = data_all.reindex(index)

    data_all = []
    for metric in metrics:
        data = pd.read_json(results_dir / f"{metric}.jsonl", orient='records', lines=True)
        data = process_data(data, datasets, [sizes[0],sizes[-1]], methods)
        # no_adaptation = data[data["method"] == "no_adaptation"]
        # min_size_no_adaptation = no_adaptation.copy()
        # min_size_no_adaptation["size"] = sizes[0]
        # max_size_no_adaptation = no_adaptation.copy()
        # max_size_no_adaptation["size"] = sizes[-1]
        data = data[data["method"] != "no_adaptation"]
        # data = pd.concat([data, min_size_no_adaptation, max_size_no_adaptation], ignore_index=True)
        data = data[data["size"].isin([sizes[0],sizes[-1]])].reset_index(drop=True)
        data["group"] = data["method"].apply(method2group)

        data["result"] = data["median"].apply(lambda x: f"{x:.2f}")
        best_idx = []
        for (dataset, size, group), group_data in data.groupby(["dataset","size","group"]):
            med = group_data["median"]
            best_idx.extend(med[med.round(2) == med.round(2).min()].index.to_list())
        # best_idx = data.groupby(["dataset","size","group"])["median"].idxmin().values
        # best_idx = data[data["median"].isin(data.loc[best_idx,"median"])].index.to_numpy()
        data.loc[best_idx,"result"] = data.loc[best_idx,:].apply(highlight_local, axis=1)
        best_idx = []
        for (dataset, size), group_data in data.groupby(["dataset","size"]):
            med = group_data["median"]
            best_idx.extend(med[med.round(2) == med.round(2).min()].index.to_list())
        # best_idx = data.groupby(["dataset","size"])["median"].idxmin().values
        # best_idx = data[data["median"].isin(data.loc[best_idx,"median"])].index.to_numpy()
        data.loc[best_idx,"result"] = data.loc[best_idx,:].apply(highlight_global, axis=1)
        data = data.pivot(index=["size","group","method"], columns="dataset", values="result")
        index = [(size, method2group(method), method) for size in [sizes[0],sizes[-1]] for method in methods]
        data = data.reindex(index)
        data = data.fillna("N/A")
        data_all.append(data)

        # for size in [sizes[0],sizes[-1]]:
        #     for method in methods:
        #         for dataset in datasets:
        #             if data_all.loc[(size, method2group(method), method), dataset] == "":
        #                 data_all.loc[(size, method2group(method), method), dataset] += data.loc[(size, method2group(method), method), dataset]
        #             else:
        #                 data_all.loc[(size, method2group(method), method), dataset] += " / " + data.loc[(size, method2group(method), method), dataset]
    
    data_all = pd.concat(data_all, axis=1, keys=metrics)
    data_all.columns = data_all.columns.swaplevel(0,1)
    data_all = data_all.loc[:,[(dataset,metric) for dataset in datasets for metric in metrics]]
    # data_all.index = data.index.map(lambda x: ({sizes[0]: f"min (N = {int(np.log2(sizes[0]))})", sizes[-1]: f"max (N = {int(np.log2(sizes[-1]))})"}[x[0]], methods_config[x[1]]["label"].replace("%","\\%").replace("\n", " &" * len(datasets) + " \\\\\n & ") ))
    smallest = f"$T' = {int(np.log2(sizes[0]))}$"
    largest = f"$T' = {int(np.log2(sizes[-1]))}$"
    data_all.index = data.index.map(lambda x: ({sizes[0]: "\\rotatebox[origin=c]{{90}}{smallest}".format(smallest="{" + smallest + "}"), sizes[-1]: "\\rotatebox[origin=c]{{90}}{largest}".format(largest="{" + largest + "}")}[x[0]], method2group(x[2]), methods_config[x[2]]["label"] ))
    data_all = data_all.reset_index(level=1,drop=True)
    data_all = data_all.loc[:,datasets]
    data_all.columns = data_all.columns.map(lambda x: (DATASETS[x[0]]["name"], metric2name[x[1]]))
    data_all.columns.name = None
    data_all.index.names = [None, None]

    # noa_data = pd.DataFrame({
    #     "dataset": datasets,
    #     "method": [methods_config["no_adaptation"]["label"]] * len(datasets),
    #     "size": [""] * len(datasets),
    #     "result": [""] * len(datasets),
    #     "metric": [""] * len(datasets),
    # }).pivot(index=["size","method"], columns=["dataset","metric"], values="result").loc[:,datasets]
    # noa_data.columns.name = None
    # noa_data.index.names = [None, None]

    noa_data = []
    for metric in metrics:
        data = pd.read_json(results_dir / f"{metric}.jsonl", orient='records', lines=True)
        data = process_data(data, datasets, [sizes[0],sizes[-1]], ["no_adaptation"])
        data["metric"] = metric
        noa_data.append(data)
        # for dataset in datasets:
        #     m = data[(data["dataset"] == dataset)]["median"].values[0]
        #     noa_data.loc[:, dataset] += f"{m:.2f} \ "
    # noa_data.iloc[0,:] = noa_data.iloc[0,:].apply(lambda x: x[:-2])
    # noa_data.columns = noa_data.columns.map(lambda x: (DATASETS[x[0]]["name"], metric2name[x[1]]))
    noa_data = pd.concat(noa_data, axis=0)
    noa_data = noa_data.pivot(index="method",columns=["dataset","metric"],values=["median"])
    noa_data.columns = noa_data.columns.droplevel(0)
    noa_data = noa_data.loc[:,[(dataset,metric) for dataset in datasets for metric in metrics]]
    noa_data.columns = noa_data.columns.map(lambda x: (DATASETS[x[0]]["name"], metric2name[x[1]]))
    noa_data.columns.names = [None, None]
    noa_data.index = noa_data.index.map(lambda x: ("",methods_config[x]["label"]))
    noa_data.index.names = ["size", "method"]
    noa_data = noa_data.apply(lambda x: x.apply(lambda y: f"{y:.2f}"), axis=1)
    data_all = pd.concat([noa_data, data_all], axis=0)

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
    table_str = table.to_latex(escape=False, column_format="ll" + "||c|c" * len(datasets))
    table_str = table_str.replace("multirow[t]", "multirow[c]")
    table_str = table_str.replace("multicolumn{2}{r}", "multicolumn{2}{c||}")
    with open(output_path, "w") as f:
        f.write(table_str)
    

    
    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)