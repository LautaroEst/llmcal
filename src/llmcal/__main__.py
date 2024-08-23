
import os
from .utils import load_yaml
from .scripts.litgpt_no_adaptation import main as litgtp_no_adaptation
from .scripts.litgpt_lora import main as litgpt_lora
from .scripts.litgpt_lora_xval import main as litgpt_lora_xval
from .scripts.litgpt_full_ft import main as litgpt_full_ft
from .scripts.bert_full_ft import main as encoder_full_ft
from .scripts.affine_calibration import main as affine_calibration, AFFINE_METHODS
from .scripts.affine_calibration_no_es import main as affine_calibration_no_es, AFFINE_METHODS_NO_ES
from .scripts.affine_calibration_train_on_val import main as affine_calibration_train_on_val, AFFINE_METHODS_TRAIN_ON_VAL
from .models import SUPPORTED_LITGPT_MODELS, SUPPORTED_ENCODER_MODELS

import datasets
datasets.disable_caching()

def main(
    dataset: str,
    prompt: str,
    model: str,
    base_method: str,
    calibration_method: str,
    **kwargs
):
    
    timing = kwargs.pop("timing", False)

    # Load config files
    dataset_config = load_yaml(f"configs/dataset/{dataset}.yaml")
    prompt_config = load_yaml(f"configs/prompt/{prompt}.yaml")
    model_config = load_yaml(f"configs/model/{model}.yaml")
    base_method_config = load_yaml(f"configs/base_method/{base_method}.yaml")
    calibration_method_config = load_yaml(f"configs/calibration_method/{calibration_method}.yaml")

    # Model name
    model_name = model_config.pop("model", None)
    if model_name is None:
        raise ValueError("Model name not found in model config")
    
    # Base method name
    base_method_name = base_method_config.pop("method", None)
    if base_method_name is None:
        raise ValueError("Base method name not found in base method config")
    
    # Calibration method name
    calibration_method_name = calibration_method_config.pop("method", None)
    if calibration_method_name is None:
        raise ValueError("Calibration method name not found in calibration method config")

    # Split kwargs into base and calibration kwargs
    calibration_kwargs = {k.split(".")[1]: v for k, v in kwargs.items() if "calibration." in k}
    for k in calibration_kwargs:
        kwargs.pop(f"calibration.{k}")
    base_kwargs = kwargs

    # Base method config
    base_config = {**dataset_config, **prompt_config, **model_config, **base_method_config}
    base_config["output_dir"] = f"experiments/{dataset}/{prompt}/{model}/{base_method}/.cache"
    base_config.update(base_kwargs)

    # Calibration method config
    calibration_config = {**dataset_config, **calibration_method_config}
    calibration_config.pop("dataset", None)
    calibration_config["output_dir"] = f"experiments/{dataset}/{prompt}/{model}/{base_method}/{calibration_method}"
    calibration_config.update(calibration_kwargs)

    # Run training method
    if base_method_name == "no_adaptation":
        litgtp_no_adaptation(**base_config, timing=timing)
    elif base_method_name == "lora" and model_name in SUPPORTED_LITGPT_MODELS:
        litgpt_lora(**base_config, timing=timing)
    elif base_method_name == "lora_xval" and model_name in SUPPORTED_LITGPT_MODELS:
        litgpt_lora_xval(**base_config)
    elif base_method_name == "full_ft" and model_name in SUPPORTED_LITGPT_MODELS:
        litgpt_full_ft(**base_config)
    elif base_method_name == "full_ft" and model_name in SUPPORTED_ENCODER_MODELS:
        encoder_full_ft(**base_config)
    else:
        raise NotImplementedError(f"Method {base_method_name} not implemented for model {model_name}")

    # Run calibration method
    if calibration_method_name == "no_calibration":
        for f in os.listdir(f"experiments/{dataset}/{prompt}/{model}/{base_method}/.cache/predictions"):
            src_pred_dir = f"../../.cache/predictions/{f}"
            dst_pred_dir = f"experiments/{dataset}/{prompt}/{model}/{base_method}/{calibration_method}/predictions/{f}"
            if not os.path.exists(dst_pred_dir):
                os.makedirs("/".join(dst_pred_dir.split("/")[:-1]), exist_ok=True)
                os.symlink(src_pred_dir, dst_pred_dir, target_is_directory=True)
    elif calibration_method_name in AFFINE_METHODS:
        affine_calibration(**calibration_config, timing=timing)
    elif calibration_method_name in AFFINE_METHODS_NO_ES:
        affine_calibration_no_es(**calibration_config)
    elif calibration_method_name in AFFINE_METHODS_TRAIN_ON_VAL:
        affine_calibration_train_on_val(**calibration_config)
    else:
        raise NotImplementedError(f"Calibration method {calibration_method_name} not implemented")
    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)