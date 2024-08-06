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

results_filename = "results_0_42_tinyllama_phi3_llama3_bert.csv"
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
# table_results = df_results.loc[df_results["model"] == "lm_phi3", ["dataset","base_method","cal_method","size","norm_cross_entropy:median", "norm_error_rate:median"]]
# table_results["method"] = (table_results["base_method"] + "+" + table_results["cal_method"]).map(method2name)
# table_results = table_results.drop(columns=["base_method", "cal_method"])
# table_results = table_results.rename(columns={"norm_cross_entropy:median":"NCE", "norm_error_rate:median":"NER"})
# table_results = table_results.loc[:,["dataset","method","size","NCE", "NER"]]
# table_results = table_results.sort_values(by=["dataset","method","size"]).set_index("dataset")
# print(table_results.to_markdown())

plot_mean_std_for_model(df_results, "lm_llama3", metrics, width = 0.5, err=False, stat="median")