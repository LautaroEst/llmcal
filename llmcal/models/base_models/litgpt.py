import os
from pathlib import Path
from typing import Optional
import torch
import lightning as L

from litgpt import GPT, Config
from litgpt.lora import GPT as LoraGPT, Config as LoraConfig, mark_only_lora_as_trainable, LoRALinear
from litgpt.utils import load_checkpoint


def init_lora_linear_modules(module):
    if isinstance(module, LoRALinear):
        module.reset_parameters()
    else:
        for child in module.children():
            init_lora_linear_modules(child)

class LoRAGPT(LoraGPT):

    @classmethod
    def from_pretrained(cls, fabric, checkpoint_dir, **kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
            if not checkpoint_dir.is_dir():
                raise ValueError(f"Invalid model_name_or_path {checkpoint_dir}")
        config = LoraConfig.from_checkpoint(checkpoint_dir, **kwargs)
        with fabric.init_module():
            model = cls(config)
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        load_checkpoint(fabric, model, checkpoint_path, strict=False)
        init_lora_linear_modules(model)
        return model

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        
        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        
        x = self.transformer.wte(idx)  # token embeddings of shape (1, T, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x) # (1, T, n_embd)
        outputs = {
            "logits": self.lm_head(x)
        }

        if output_last_hidden_state:
            outputs["last_hidden_state"] = x
            
        return outputs

class LitGPT(GPT):

    @classmethod
    def from_pretrained(cls, fabric, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
            if not checkpoint_dir.is_dir():
                raise ValueError(f"Invalid model_name_or_path {checkpoint_dir}")
        config = Config.from_checkpoint(checkpoint_dir)
        with fabric.init_module():
            model = cls(config)
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        if not checkpoint_path.is_file():
            checkpoint_path = checkpoint_dir / "checkpoint.ckpt"
            if not checkpoint_path.is_file():
                raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        load_checkpoint(fabric, model, checkpoint_path, strict=False)
        for param in model.parameters():
            param.requires_grad = True
        return model

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        
        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        
        x = self.transformer.wte(idx)  # token embeddings of shape (1, T, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x) # (1, T, n_embd)
        outputs = {
            "logits": self.lm_head(x)
        }

        if output_last_hidden_state:
            outputs["last_hidden_state"] = x
            
        return outputs
    

class LoRAGPTForClassification(LoraGPT):

    @classmethod
    def from_pretrained(cls, fabric, checkpoint_dir, **kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
            if not checkpoint_dir.is_dir():
                raise ValueError(f"Invalid model_name_or_path {checkpoint_dir}")
        num_classes = kwargs.pop("num_classes", 2)
        config = LoraConfig.from_checkpoint(checkpoint_dir, **kwargs)
        with fabric.init_module():
            model = cls(config)
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        load_checkpoint(fabric, model, checkpoint_path, strict=False)
        init_lora_linear_modules(model)
        del model.lm_head
        model.classifier = torch.nn.Linear(model.transformer.n_embd, num_classes)
        return model

    def _forward_single_sample(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        
        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        
        x = self.transformer.wte(idx)  # token embeddings of shape (1, T, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x) # (1, T, n_embd)
        outputs = {
            "logits": self.classifier(x[:,-1,:])
        }

        if output_last_hidden_state:
            outputs["last_hidden_state"] = x
            
        return outputs
    
    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        logits = []
        for input_ids, attention_mask, answers in zip(input_ids, attention_mask):
            T = torch.sum(attention_mask)
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            logits.append(self._forward_single_sample(input_ids)["logits"])
        logits = torch.stack(logits, dim=0)
        return {"logits": logits}

class LitGPTForClassification(GPT):

    @classmethod
    def from_pretrained(cls, fabric, checkpoint_dir, num_classes = 2):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
            if not checkpoint_dir.is_dir():
                raise ValueError(f"Invalid model_name_or_path {checkpoint_dir}")
        config = Config.from_checkpoint(checkpoint_dir)
        with fabric.init_module():
            model = cls(config)
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        load_checkpoint(fabric, model, checkpoint_path, strict=False)
        for param in model.parameters():
            param.requires_grad = True
        del model.lm_head
        model.classifier = torch.nn.Linear(model.transformer.n_embd, num_classes)
        return model

    def _forward_single_sample(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        
        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        
        x = self.transformer.wte(idx)  # token embeddings of shape (1, T, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x) # (1, T, n_embd)
        outputs = {
            "logits": self.classifier(x[:,-1,:])
        }

        if output_last_hidden_state:
            outputs["last_hidden_state"] = x
            
        return outputs
    
    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        logits = []
        for input_ids, attention_mask, answers in zip(input_ids, attention_mask):
            T = torch.sum(attention_mask)
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            logits.append(self._forward_single_sample(input_ids)["logits"])
        logits = torch.stack(logits, dim=0)
        return {"logits": logits}