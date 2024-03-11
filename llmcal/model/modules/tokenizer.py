import os
from pathlib import Path
from typing import Dict, List, Literal

import torch
from lit_gpt import Tokenizer as _Tokenizer
from lit_gpt.utils import check_valid_checkpoint_dir

class LitGPTTokenizer:

    def __init__(self, model_name_or_path: str):
        super().__init__()
        model_name_or_path = Path(model_name_or_path)
        if not model_name_or_path.is_dir():
            model_name_or_path = Path(os.getenv("LIT_CHECKPOINTS")) / model_name_or_path
            if not model_name_or_path.is_dir():
                raise ValueError(f"Invalid model_name_or_path {model_name_or_path}")
        check_valid_checkpoint_dir(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = _Tokenizer(model_name_or_path)
        self.tokenizer.pad_token_id = 0

    def __call__(self, prompts: List[str]) -> Dict[Literal["input_ids","attention_mask"], torch.LongTensor]:
        device = torch.device("cpu")
        input_ids = []
        lens = []
        for prompt in prompts:
            idx = self.tokenizer.encode(prompt, device=device, bos=True)
            input_ids.append(idx)
            lens.append(len(idx))
        max_len = max(lens)

        padded_input_ids = []
        padded_attention_mask = []
        for idx in input_ids:
            padded_input_ids.append(
                torch.cat([torch.zeros(max_len - len(idx), dtype=torch.long, device=device), idx])
            )
            padded_attention_mask.append(
                torch.cat([
                    torch.zeros(max_len - len(idx), dtype=torch.long, device=device), 
                    torch.ones(len(idx), dtype=torch.long, device=device)
                ])
            )
        return {
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "attention_mask": torch.stack(padded_attention_mask, dim=0)
        }