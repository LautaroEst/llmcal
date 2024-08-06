
from collections import OrderedDict
import pandas as pd

methods = OrderedDict([
    # ("no_adaptation+temp_scaling", "Scale-only Calibration\n(Temperature Scaling)"),
    # ("no_adaptation+bias_only", "Bias-only Calibration"),
    ("no_adaptation+affine_scalar", "DP Calibration"),
    ("lora+affine_scalar_train_on_val", "LoRA + DP Calibration"),
    ("lora+no_calibration", "LoRA"),
    ("full_ft+no_calibration", "Fine-Tunning"),
])


def main():
    # results_filename = "results_0_42_paper.csv"
    results_filename = "results_0_42_tinyllama_phi3_llama3_bert.csv"
    # skip_datasets = ["20newsgroups", "banking77"]
    skip_datasets = []
    metric = "norm_cross_entropy"
    results = pd.read_csv(results_filename, header=0, index_col=None)
    results = results.loc[(results["split"] == "test"),["model", "dataset","size", "base_method", "cal_method", f"{metric}:median"]]
    results["method"] = results["base_method"] + "+" + results["cal_method"]
    results = results.drop(columns=["base_method", "cal_method"])
    new_results = []
    # for model in results["model"].unique():
    for model in ["lm_tinyllama", "roberta_base"]:
        model_results = results.loc[results["model"] == model]
        for dataset in model_results["dataset"].unique():
            if dataset in skip_datasets:
                continue
            dataset_results = model_results.loc[model_results["dataset"] == dataset]
            small_size = dataset_results["size"].min()
            large_size = dataset_results["size"].max()
            for name, size in zip(["small", "large"], [small_size, large_size]):
                # lora_value = dataset_results.loc[(dataset_results["size"] == size) & (dataset_results["method"] == "lora+no_calibration"), f"{metric}:median"].item()
                for method in dataset_results.loc[dataset_results["size"] == size, "method"].values:
                    value = dataset_results.loc[(dataset_results["size"] == size) & (dataset_results["method"] == method), f"{metric}:median"].item()
                    new_results.append({
                        "model": model,
                        "dataset": dataset,
                        "size": name,
                        "method": method,
                        f"{metric}:value:median": value,
                        # f"{metric}:loranorm:median": (value - lora_value) / value,
                    })
                    # value = results.loc[(
                    #     (results["size"] == size) & \
                    #     (results["dataset"] == dataset) & \
                    #     (results["method"] == "full_ft+no_calibration") & \
                    #     (results["model"] == "roberta_base")
                    # ), f"{metric}:median"].item()
                    # new_results.append({
                    #     "model": "roberta_base",
                    #     "dataset": dataset,
                    #     "size": name,
                    #     "method": "full_ft+no_calibration",
                    #     f"{metric}:value:median": value,
                    # })
                
    results = pd.DataFrame(new_results)
    results = results.groupby(["model","method","size"]).agg({f"{metric}:value:median": "mean"}).reset_index()
    results["method"] = results["method"].map(methods)
    results["model"] = results["model"].map({
        "lm_phi3": "Phi-3",
        "lm_tinyllama": "TinyLlama",
        "lm_llama3": "Llama-3",
        "roberta_base": "RoBERTa",
    })
    results = results.loc[results["method"].isin(methods.values())]
    results = results.pivot(index=["model","method"], columns="size", values=f"{metric}:value:median").loc[:,["small","large"]].round(2)
    # results = results.loc[["TinyLlama", "Llama-3", "RoBERTa"],:].reset_index().set_index("model")
    results = results.reset_index().set_index("model")

    # keep lower and greater size
    print(results.to_markdown())

if __name__ == "__main__":
    main()