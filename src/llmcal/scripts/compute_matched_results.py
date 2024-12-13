
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..src.evaluation.metrics import compute_psr_with_mincal


METHODS = [
    "no_adaptation",
    "dpcal",
    "tempscaling",
    "finetunne_lora",
    "lora_plus_dpcal",
    "lora_plus_tempscaling",
]


def read_finetuning_results(root_results_dir: Path):
    data = []
    for dataset_dir in root_results_dir.iterdir():
        train_dataset = dataset_dir.name
        for size_dir in dataset_dir.iterdir():
            size = int(size_dir.name.split("=")[1])
            for seed_dir in size_dir.iterdir():
                seed = int(seed_dir.name.split("=")[1])
                for method in seed_dir.iterdir():
                    method_name = method.name
                    for train_lst in method.iterdir():
                        train_lst_name = train_lst.name
                        for val_lst in train_lst.iterdir():
                            val_lst_name = val_lst.name
                            for test_dataset_dir in val_lst.iterdir():
                                if not test_dataset_dir.name.startswith("test="):
                                    continue
                                test_dataset = test_dataset_dir.name.split("=")[1]
                                for test_lst in test_dataset_dir.iterdir():
                                    if not test_lst.name.startswith("list=test"):
                                        continue
                                    test_lst_name = test_lst.name.split("=")[1]
                                    if not (logits_path := test_lst / "logits.csv").exists():
                                        continue
                                    if not (labels_path := test_lst / "labels.csv").exists():
                                        continue
                                    data.append({
                                        "train_dataset": train_dataset,
                                        "size": size,
                                        "seed": seed,
                                        "method": method_name,
                                        "train_lst": train_lst_name,
                                        "val_lst": val_lst_name,
                                        "cal_lst": None,
                                        "test_dataset": test_dataset,
                                        "test_lst": test_lst_name,
                                        "logits": logits_path,
                                        "labels": labels_path,
                                    })

    return pd.DataFrame(data)

def read_lora_plus_calibration_results(root_results_dir: Path):
    data = []
    for dataset_dir in root_results_dir.iterdir():
        train_dataset = dataset_dir.name
        for size_dir in dataset_dir.iterdir():
            size = int(size_dir.name.split("=")[1])
            for seed_dir in size_dir.iterdir():
                seed = int(seed_dir.name.split("=")[1])
                for method in seed_dir.iterdir():
                    method_name = method.name
                    for train_lst in method.iterdir():
                        train_lst_name = train_lst.name
                        for val_lst in train_lst.iterdir():
                            val_lst_name = val_lst.name
                            for cal_lst in val_lst.iterdir():
                                cal_lst_name = cal_lst.name
                                for test_dataset_dir in cal_lst.iterdir():
                                    if not test_dataset_dir.name.startswith("test="):
                                        continue
                                    test_dataset = test_dataset_dir.name.split("=")[1]
                                    for test_lst in test_dataset_dir.iterdir():
                                        if not test_lst.name.startswith("list=test"):
                                            continue
                                        test_lst_name = test_lst.name.split("=")[1]
                                        if not (logits_path := test_lst / "logits.csv").exists():
                                            continue
                                        if not (labels_path := test_lst / "labels.csv").exists():
                                            continue
                                        data.append({
                                            "train_dataset": train_dataset,
                                            "size": size,
                                            "seed": seed,
                                            "method": method_name,
                                            "train_lst": train_lst_name,
                                            "val_lst": val_lst_name,
                                            "cal_lst": cal_lst_name,
                                            "test_dataset": test_dataset,
                                            "test_lst": test_lst_name,
                                            "logits": logits_path,
                                            "labels": labels_path,
                                        })

    return pd.DataFrame(data)
                        


def read_calibration_results(root_results_dir: Path):
    data = []
    for dataset_dir in root_results_dir.iterdir():
        train_dataset = dataset_dir.name
        for size_dir in dataset_dir.iterdir():
            size = int(size_dir.name.split("=")[1])
            for seed_dir in size_dir.iterdir():
                seed = int(seed_dir.name.split("=")[1])
                for method in seed_dir.iterdir():
                    method_name = method.name
                    for train_lst in method.iterdir():
                        train_lst_name = train_lst.name
                        for val_lst in train_lst.iterdir():
                            val_lst_name = val_lst.name
                            for test_dataset_dir in val_lst.iterdir():
                                if not test_dataset_dir.name.startswith("test="):
                                    continue
                                test_dataset = test_dataset_dir.name.split("=")[1]
                                for test_lst in test_dataset_dir.iterdir():
                                    if not test_lst.name.startswith("list=test"):
                                        continue
                                    test_lst_name = test_lst.name.split("=")[1]
                                    if not (logits_path := test_lst / "logits.csv").exists():
                                        continue
                                    if not (labels_path := test_lst / "labels.csv").exists():
                                        continue
                                    data.append({
                                        "train_dataset": train_dataset,
                                        "size": size,
                                        "seed": seed,
                                        "method": method_name,
                                        "train_lst": train_lst_name,
                                        "val_lst": val_lst_name,
                                        "cal_lst": None,
                                        "test_dataset": test_dataset,
                                        "test_lst": test_lst_name,
                                        "logits": logits_path,
                                        "labels": labels_path,
                                    })

    return pd.DataFrame(data)

def compute_metrics(data, metric):
    data_with_metrics = data.copy()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        logits = pd.read_csv(row["logits"], index_col=0, header=None).values.astype(float)
        labels = pd.read_csv(row["labels"], index_col=0, header=None).values.flatten().astype(int)
        value, min_value = compute_psr_with_mincal(logits, labels, metric, "none")
        data_with_metrics.loc[i, "result"] = value
        data_with_metrics.loc[i, "min_result"] = min_value
    data_with_metrics = data_with_metrics.drop(columns=["logits", "labels"])
    return data_with_metrics


def extract_method(row):

    if row["method_type"] == "no_adaptation":
        method = row["method_type"]
    elif row["method_type"] == "calibration":
        method = row["method"]
    elif row["method_type"] == "finetune_lora":
        s, e = map(float,row["train_lst"].split("-"))
        p_train = e - s
        if row["method"] != "lora_ans_no_es":
            method = f"lora_{p_train:.1f}"
        else:
            method = f"lora_{p_train:.1f}_no_es"
    elif row["method_type"] in ["lora_plus_dpcal", "lora_plus_tempscaling", "lora_plus_dpcal_trainontest", "lora_plus_tempscaling_trainontest"]:
        s, e = map(float,row["train_lst"].split("-"))
        p_train = e - s
        if row["method"] != "lora_ans_no_es":
            method = f"lora_{p_train:.1f}" + "_plus_" + row["method_type"].split("_plus_")[1]
        else:
            method = f"lora_{p_train:.1f}_no_es" + "_plus_" + row["method_type"].split("_plus_")[1]
    else:
        raise ValueError(f"Unknown method: {row['method_type']}, {row['method']}")
    
    return method



def process_data(data, reduced = False):

    # Keep matched trainings
    data = data[data["train_dataset"] == data["test_dataset"]]
    data = data.drop(columns=["train_dataset"])
    data = data.rename(columns={"test_dataset": "dataset"})

    # Keep evaluation in test
    if reduced:
        data = data[data["test_lst"].str.startswith("test_")]
    else:
        data = data[data["test_lst"] == "test"]
    data = data.drop(columns=["test_lst"])

    # Replace method name for full description
    data["method"] = data.apply(extract_method, axis=1)
    data = data.drop(columns=["method_type", "train_lst", "val_lst", "cal_lst"])

    # Reorder columns
    data = data.loc[:, ["dataset", "method", "size", "seed", "result", "min_result"]]

    return data



def main(
    metric: str,
    finetuning_root_results_dirs: str,
    lora_plus_cal_root_results_dirs: str,
    cal_root_results_dirs: str,
    trainontest_root_results_dirs: str,
    output_path: str,
    reduced: bool = False,
):
    # Read results
    finetuning_root_results_dirs = [Path(d) for d in finetuning_root_results_dirs.split(",")] if finetuning_root_results_dirs is not None else []
    cal_root_results_dirs = [Path(d) for d in cal_root_results_dirs.split(",")] if cal_root_results_dirs is not None else []
    lora_plus_cal_root_results_dirs = [Path(d) for d in lora_plus_cal_root_results_dirs.split(",")] if lora_plus_cal_root_results_dirs is not None else []
    trainontest_root_results_dirs = [Path(d) for d in trainontest_root_results_dirs.split(",")] if trainontest_root_results_dirs is not None else []
    all_data = []
    for root_results_dir in finetuning_root_results_dirs:
        finetuning_data = read_finetuning_results(root_results_dir)
        finetuning_data["method_type"] = str(root_results_dir).split("/")[-2]
        all_data.append(finetuning_data)
    for root_results_dir in cal_root_results_dirs:
        cal_data = read_calibration_results(root_results_dir)
        cal_data["method_type"] = str(root_results_dir).split("/")[-2]
        all_data.append(cal_data)
    for root_results_dir in lora_plus_cal_root_results_dirs:
        cal_data = read_lora_plus_calibration_results(root_results_dir)
        cal_data["method_type"] = str(root_results_dir).split("/")[-2]
        all_data.append(cal_data)
    for root_results_dir in trainontest_root_results_dirs:
        trainontest_data = read_lora_plus_calibration_results(root_results_dir)
        trainontest_data["method_type"] = str(root_results_dir).split("/")[-2]
        all_data.append(trainontest_data)

    data = pd.concat(all_data, ignore_index=True)
    
    # Compute metrics
    data_with_metrics = compute_metrics(data, metric)

    # Process data
    data_with_metrics = process_data(data_with_metrics, reduced)

    # Save data
    data_with_metrics.to_json(output_path, orient="records", lines=True)
    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)