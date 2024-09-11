from results_utils import compute_results, plot_mean_std_for_model, load_results_paths, compute_metric, boxplot_for_model, plot_manual_boxplot_for_model, supported_methods
import os
import pandas as pd

method2name = {
    "no_adaptation+no_calibration": "No Adaptation",
    "no_adaptation+affine_scalar": "DP Calibration",
    "no_adaptation+bias_only": "Bias-only Calibration",
    "no_adaptation+temp_scaling": "Scale-only Calibration",
    "lora+no_calibration": "LoRA",
    "lora+affine_scalar_train_on_val": "LoRA + DP Calibration",
}

metrics = ["norm_error_rate", "norm_cross_entropy", "ece"]#, "cal_loss_bias", "norm_min_calibration_bias"]
# metrics = ["norm_error_rate", "norm_cross_entropy"]

# results_filename = "results_0_42.csv"
# results_filename = "results_0_42_tinyllama_phi3_llama3_bert.csv"
# results_filename = "results_0_42_phi3.csv"
# results_filename = "results_0_42_tinyllama.csv"
# results_filename = "results_llama3_03092024.csv"
# results_filename = "results_phi3_03092024.csv"
# results_filename = "results_tinyllama_03092024.csv"
# results_filename = "results_tinyllama_03092024_mahalanobis.csv"
# results_filename = "results_phi3_10092024.csv"
results_filename = "results_llama3_bert_11092024.csv"
bootstrap = 0
random_state = 42
if os.path.exists(results_filename):
    df = pd.read_csv(results_filename, index_col=False)
else:
    df = compute_results(metrics, bootstrap, random_state)
    df.to_csv(results_filename, index=False)
# df = compute_results(metrics, bootstrap, random_state)
df.loc[:,[c for c in df.columns if ":" not in c ]]

df_results = df
# table_results = df_results.loc[df_results["model"] == "lm_llama3", ["dataset","base_method","cal_method","size","norm_cross_entropy:median", "norm_error_rate:median"]]
# table_results["method"] = (table_results["base_method"] + "+" + table_results["cal_method"]).map(method2name)
# table_results = table_results.drop(columns=["base_method", "cal_method"])
# table_results = table_results.rename(columns={"norm_cross_entropy:median":"NCE", "norm_error_rate:median":"NER"})
# table_results = table_results.loc[:,["dataset","method","size","NCE", "NER"]]
# table_results = table_results.sort_values(by=["dataset","method","size"])

# new_table_results = []
# grouped = table_results.groupby("dataset")
# for dataset, dataset_group in grouped:
#     dataset_first = True
#     method_grouped = dataset_group.groupby("method")
#     for method, method_group in method_grouped:
#         method_first = True
#         for _, row in method_group.iterrows():
#             if dataset_first and method_first:
#                 new_table_results.append([dataset, method, row["size"], row["NCE"], row["NER"]])
#                 dataset_first = False
#                 method_first = False
#             elif method_first and not dataset_first:
#                 new_table_results.append(['', method, row["size"], row["NCE"], row["NER"]])
#                 method_first = False
#             else:
#                 new_table_results.append(['', '', row["size"], row["NCE"], row["NER"]])
# new_table_results = pd.DataFrame(new_table_results, columns=table_results.columns).set_index("dataset")
# print(new_table_results.to_markdown())

# plot_mean_std_for_model(df_results, "lm_tinyllama", metrics, width = 0.5, err=False, stat="median")
# table_results = df_results.loc[df_results["model"] == "lm_phi3", ["dataset","base_method","cal_method","size","norm_cross_entropy:median", "norm_error_rate:median"]]
# table_results["method"] = (table_results["base_method"] + "+" + table_results["cal_method"]).map(method2name)
# table_results = table_results.drop(columns=["base_method", "cal_method"])
# table_results = table_results.rename(columns={"norm_cross_entropy:median":"NCE", "norm_error_rate:median":"NER"})
# table_results = table_results.loc[:,["dataset","method","size","NCE", "NER"]]
# table_results = table_results.sort_values(by=["dataset","method","size"]).set_index("dataset")
# print(table_results.to_markdown())

plot_mean_std_for_model(df_results, "lm_llama3", metrics, width = 0.5, err=False, stat="median")
