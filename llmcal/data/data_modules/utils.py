from pathlib import Path
from typing import Dict, List, Literal, Optional
import torch
from litgpt import Tokenizer as _Tokenizer
from litgpt.utils import check_valid_checkpoint_dir


class LitGPTTokenizer:

    def __init__(self, checkpoint_dir: Path):
        check_valid_checkpoint_dir(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = _Tokenizer(checkpoint_dir)
        self.pad_token_id = 0

    def __call__(self, prompts: List[str], max_seq_length: Optional[int] = None) -> Dict[Literal["input_ids","attention_mask"], torch.LongTensor]:
        device = torch.device("cpu")
        input_ids = []
        lens = []
        for prompt in prompts:
            idx = self.tokenizer.encode(prompt, device=device, bos=True)
            if max_seq_length:
                idx = idx[:max_seq_length]
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

    def __init__(self, pad_token_id, max_seq_len):
        # batch = {"idx": ..., "prompt_ids": ..., "answers_ids": ...}
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        prompts_ids = []
        prompt_masks = []
        answers_ids = []
        max_ans_len = max([max([ans.shape[0] for ans in sample["answers_ids"]]) for sample in batch])
        max_prompt_len = min(self.max_seq_len - max_ans_len, max([sample["prompt_ids"].shape[0] for sample in batch]))
        for sample in batch:
            prompts_ids.append(torch.cat([torch.ones(max_prompt_len - sample["prompt_ids"].shape[0], dtype=torch.long) * self.pad_token_id, sample["prompt_ids"]]))
            prompt_masks.append(torch.cat([torch.zeros(max_prompt_len - sample["prompt_ids"].shape[0], dtype=torch.long), torch.ones(sample["prompt_ids"].shape[0], dtype=torch.long)]))
            answers_ids.append(sample["answers_ids"])
        return {
            "idx": torch.stack([sample["idx"] for sample in batch]),
            "prompt_ids": torch.stack(prompts_ids),
            "prompt_mask": torch.stack(prompt_masks),
            "answers_ids": answers_ids,
            "label": torch.stack([sample["label"] for sample in batch])
        }