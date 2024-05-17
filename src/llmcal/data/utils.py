import torch


class LitGPTCollator:

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
            seq = sample["prompt_ids"][-max_prompt_len:]
            prompts_ids.append(torch.cat([torch.ones(max_prompt_len - seq.shape[0], dtype=torch.long) * self.pad_token_id, seq]))
            prompt_masks.append(torch.cat([torch.zeros(max_prompt_len - seq.shape[0], dtype=torch.long), torch.ones(seq.shape[0], dtype=torch.long)]))
            answers_ids.append(sample["answers_ids"])
        return {
            "idx": torch.stack([sample["idx"] for sample in batch]),
            "prompt_ids": torch.stack(prompts_ids),
            "prompt_mask": torch.stack(prompt_masks),
            "answers_ids": answers_ids,
            "label": torch.stack([sample["label"] for sample in batch])
        }