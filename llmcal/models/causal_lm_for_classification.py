import os
from pathlib import Path
from typing import Dict, Literal, Optional
import torch
import lightning as L

from litgpt import GPT, Config
from litgpt.lora import GPT as LoraGPT, Config as LoraConfig, mark_only_lora_as_trainable, LoRALinear
from litgpt.utils import load_checkpoint




class CausalLMForClassification(L.LightningModule):

    def __init__(
        self, 
        language_model,
        loss_fn: Literal["cross_entropy"] = "cross_entropy",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 2,
        micro_batch_size: int = 1
    ):
        super().__init__()
        self.lm = language_model
        self.lm.set_kv_cache(batch_size=1)

        # Training arguments
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.global_batch_size = batch_size
        self.micro_batch_size = micro_batch_size

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        return self.lm(idx, input_pos, output_last_hidden_state)
    
    def on_train_start(self) -> None:
        self.gradient_accumulation_steps = (self.global_batch_size // self.fabric.world_size) // self.micro_batch_size
        if self.loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {self.loss_fn} not implemented")
        self.train()
    
    def training_step(
        self,
        batch: Dict[Literal["prompt_ids", "prompt_mask", "answers_ids", "labels"], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        answers_ids = batch["answers_ids"]
        labels = batch["labels"]

        loss = 0
        num_tokens = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = torch.cat([
                input_ids[attention_mask == 1].unsqueeze(0),
                answers[label.item()].unsqueeze(0)
            ], dim=1)
            logprobs = self(input_ids)["logits"][:,:-1,:].log_softmax(dim=2)
            index = input_ids[:,1:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + input_ids.shape[1]
        loss = loss / num_tokens
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.lm.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def on_validation_start(self) -> None:
        self.eval()

    def validation_step(
        self,
        batch: Dict[Literal["prompt_ids", "prompt_mask", "answers_ids", "labels"], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        answers_ids = batch["answers_ids"]
        labels = batch["labels"]

        loss = 0
        num_tokens = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = torch.cat([
                input_ids[attention_mask == 1].unsqueeze(0),
                answers[label.item()].unsqueeze(0)
            ], dim=1)
            logprobs = self(input_ids)["logits"][:,:-1,:].log_softmax(dim=2)
            index = input_ids[:,1:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + input_ids.shape[1]
        return {"cum_loss": loss, "num_tokens": num_tokens}
    
