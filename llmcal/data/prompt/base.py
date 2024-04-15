

import os
import re
from typing import Dict, List, Literal, Optional
import torch
from transformers import AutoTokenizer
from litgpt import Tokenizer as _Tokenizer

class Tokenizer:

    def __init__(self, model_name_or_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.tokenizer = tokenizer
            self.is_hf = True
        except Exception:
            if not os.path.exists(model_name_or_path):
                model_name_or_path = os.path.join(os.getenv("LIT_CHECKPOINTS"), model_name_or_path)
                if not os.path.exists(model_name_or_path):
                    raise ValueError(f"Tokenizer directory {model_name_or_path} does not exist and  could not load tokenizer from {model_name_or_path} in huggingface hub.")
                self.tokenizer = _Tokenizer(model_name_or_path)
                self.tokenizer.pad_token_id = 0
            self.is_hf = False
        self.model_name_or_path = model_name_or_path
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, prompts: List[str], max_seq_length: Optional[int] = None, bos: Optional[bool] = True) -> Dict[Literal["input_ids","attention_mask"], torch.LongTensor]:
        if self.is_hf:
            return self.tokenizer(prompts, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")

        device = torch.device("cpu")
        input_ids = []
        lens = []
        for prompt in prompts:
            idx = self.tokenizer.encode(prompt, device=device, bos=True)
            if max_seq_length:
                idx = idx[:max_seq_length]
            if not bos:
                idx = idx[1:]
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

class DynamicPaddingCollator:

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        device = torch.device("cpu")
        input_ids = [item["prompt_ids"] for item in batch]
        max_len = max([len(ids) for ids in input_ids])
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
            "input_ids": torch.cat(padded_input_ids, dim=0),
            "attention_mask": torch.cat(padded_attention_mask, dim=0)
        }

class Prompt:

    def __init__(self, prompt_template, tokenizer_dir, max_seq_len, answers_templates=None):
        self.prompt_template = prompt_template
        self.answers_templates = answers_templates
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer(tokenizer_dir)            
        self.prompt_features = re.findall(r'\{(\w+)\}', prompt_template)

            
    def fill_and_tokenize(self, **kwargs):
        if not set(self.prompt_features).issubset(set(kwargs.keys())):
            raise ValueError(f"Expected features {self.prompt_features} in kwargs, got {list(kwargs.keys())}")
        prompt = self.prompt_template.format(**{k: v for k, v in kwargs.items() if k in self.prompt_features})
        
        if self.answers_templates is None:
            return {"prompt_ids": self.tokenizer([prompt], max_length=self.max_seq_len)["input_ids"]}

        answers_ids = [self.tokenizer([answer.format(**{k: v for k, v in kwargs.items() if k in re.findall(r'\{(\w+)\}', answer)})], bos=False)["input_ids"][0] for answer in self.answers_templates]
        return {
            "prompt_ids": self.tokenizer([prompt], max_seq_length=self.max_seq_len)["input_ids"],
            "answers_ids": answers_ids,
        }
        