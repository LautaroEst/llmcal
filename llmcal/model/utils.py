import os
from transformers import AutoConfig
from typing import Literal
from pathlib import Path
from litgpt.utils import check_valid_checkpoint_dir

def check_model_type(checkpoint_dir: str, method: Literal["no_adaptation", "full_ft", "lora", "affine_calibration"]) -> Literal["litgpt", "hf"]:

    if not os.path.exists(checkpoint_dir):
        if not os.path.exists(Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir):
            try:
                AutoConfig.from_pretrained(checkpoint_dir)
                model_type = "hf"
            except:
                raise ValueError(f"Invalid checkpoint directory: {checkpoint_dir}")
        else:
            check_valid_checkpoint_dir(Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir, lora = False)
            model_type = "litgpt"
    else:
        check_valid_checkpoint_dir(Path(checkpoint_dir), lora = False)
        model_type = "litgpt"

    return model_type