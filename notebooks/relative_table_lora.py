
from collections import OrderedDict
import pandas as pd

methods = OrderedDict([
    ("no_adaptation+temp_scaling", "Scale-only Calibration\n(Temperature Scaling)"),
    ("no_adaptation+bias_only", "Bias-only Calibration"),
    ("no_adaptation+affine_scalar", "DP Calibration"),
    ("lora+affine_scalar_train_on_val", "LoRA + DP Calibration"),
])


def main():
    # results_filename = "results_0_42_paper.csv"
    results_filename = "results_0_42.csv"
    skip_datasets = ["20newsgroups"]
    metric = "norm_cross_entropy"
    results = pd.read_csv(results_filename, header=0, index_col=None)
    results = results.loc[results["split"] == "test",["dataset","size", "base_method", "cal_method", f"{metric}:median"]]
    results["method"] = results["base_method"] + "+" + results["cal_method"]
    results = results.drop(columns=["base_method", "cal_method"])
    new_results = []
    for dataset in results["dataset"].unique():
        if dataset in skip_datasets:
            continue
        dataset_results = results.loc[results["dataset"] == dataset]
        min_size = dataset_results["size"].min()
        max_size = dataset_results["size"].max()
        for name, size in zip(["min", "max"], [min_size, max_size]):
            lora_value = dataset_results.loc[(dataset_results["size"] == size) & (dataset_results["method"] == "lora+no_calibration"), f"{metric}:median"].item()
            for method in dataset_results.loc[dataset_results["size"] == size, "method"].values:
                value = dataset_results.loc[(dataset_results["size"] == size) & (dataset_results["method"] == method), f"{metric}:median"].item()
                new_results.append({
                    "dataset": dataset,
                    "size": name,
                    "method": method,
                    f"{metric}:value:median": value,
                    f"{metric}:loranorm:median": (value - lora_value) / value,
                })
    results = pd.DataFrame(new_results)
    results = results.groupby(["method","size"]).agg({"norm_cross_entropy:loranorm:median": "mean"}).reset_index()
    results = results.pivot(index="method", columns="size", values="norm_cross_entropy:loranorm:median").loc[:,["min","max"]]
    results.index = results.index.map(methods)
    results = results.loc[methods.values()]
    results = (results * 100).round(2)

    # keep lower and greater size
    print(results.to_markdown())

if __name__ == "__main__":
    main()