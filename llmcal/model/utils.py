import os
from transformers import AutoConfig
from litgpt import Config
from typing import Literal
from pathlib import Path

def check_model_type(checkpoint_dir: str) -> Literal["litgpt", "hf"]:
    model_type = None
    try:
        Config.from_checkpoint(Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir)
        model_type = "litgpt"
    except:
        try:
            AutoConfig.from_pretrained(checkpoint_dir)
            model_type = "hf"
        except:
            pass

    if model_type is not None:
        return model_type
    
    raise ValueError(f"Invalid model type: {checkpoint_dir}")