
import os
from pathlib import Path
from typing import List, Literal, Optional
import torch
from torch import nn
import lightning as L

from lit_gpt.lora import GPT, Config, mark_only_lora_as_trainable
from lit_gpt.utils import load_checkpoint
from .tokenizer import LitGPTTokenizer


class LoRALitGPT(GPT):

    def __init__(self, model_name_or_path: str, **lora_kwargs):
        self.model_name_or_path = model_name_or_path
        model_name_or_path = Path(model_name_or_path)
        if not model_name_or_path.is_dir():
            model_name_or_path = Path(os.getenv("LIT_CHECKPOINTS")) / model_name_or_path
            if not model_name_or_path.is_dir():
                raise ValueError(f"Invalid model_name_or_path {model_name_or_path}")
        config = Config.from_checkpoint(model_name_or_path)
        for k, v in lora_kwargs.items():
            setattr(config, k.split("lora_")[-1], v)
        super().__init__(config)
        self.set_kv_cache(batch_size=1)
        self.tokenizer = LitGPTTokenizer(model_name_or_path)

    def init_params(self, fabric: L.Fabric):
        mark_only_lora_as_trainable(self)
        checkpoint_path = Path(self.model_name_or_path) / "lit_model.pth"
        if not checkpoint_path.is_file():
            checkpoint_path = Path(os.getenv("LIT_CHECKPOINTS")) / self.model_name_or_path / "lit_model.pth"
        load_checkpoint(fabric, self, checkpoint_path, strict=False)

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            "last_hidden_state": x,
            "logits": self.lm_head(x)
        }
        return outputs

class LoRALitGPTLanguageModel(LoRALitGPT):

    def forward(
        self, 
        prompt_ids: torch.Tensor, 
        prompt_mask: torch.Tensor, 
    ):
        outputs = {"last_hidden_state": [], "logits": []}
        for input_ids, attention_mask in zip(prompt_ids, prompt_mask):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            output = super().forward(input_ids)
            outputs["last_hidden_state"].append(
                torch.cat([torch.zeros(1, (attention_mask == 0).sum(), self.config.n_embd, dtype=output["last_hidden_state"].dtype, device=output["last_hidden_state"].device), output["last_hidden_state"]],dim=1)
            )
            outputs["logits"].append(
                torch.cat([torch.zeros(1, (attention_mask == 0).sum(), self.config.vocab_size, dtype=output["logits"].dtype, device=output["logits"].device), output["logits"]],dim=1)
            )
        outputs["last_hidden_state"] = torch.cat(outputs["last_hidden_state"], dim=0)
        outputs["logits"] = torch.cat(outputs["logits"], dim=0)
        return outputs

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: str = "cross_entropy",
    ):
        if loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")
        
        outputs = self(prompt_ids, prompt_mask)["logits"].log_softmax(dim=2)
        gather_lobprobs = torch.gather(outputs, -1, labels.unsqueeze(2)).squeeze(2)
        loss = -(gather_lobprobs * prompt_mask).sum() / torch.sum(prompt_mask)
        return loss
    
    def predict_step(
        self,
        prompt_ids: torch.Tensor, 
        prompt_mask: torch.Tensor,
    ):
        return self(prompt_ids, prompt_mask)
        

                
class LoRALitGPTPromptClassification(LoRALitGPT):

    def __init__(self, model_name_or_path: str, embedding_pooling: Literal["mean", "max", "last"] = "last", **lora_kwargs):
        super().__init__(model_name_or_path, **lora_kwargs)
        self.embedding_pooling = embedding_pooling

    def predict_step(
        self,
        prompt_ids: torch.Tensor, 
        prompt_mask: torch.Tensor, 
        answers_ids: List[List[torch.Tensor]],
    ):
        prompt_hidden_states = []
        logits = []
        for input_ids, attention_mask, answers in zip(prompt_ids, prompt_mask, answers_ids):
            T = torch.sum(attention_mask)
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            output = self(input_ids)
            answers_logits = []
            for answer in answers:
                answer = answer.unsqueeze(0)
                input_pos = torch.arange(T, answer.shape[1] + T, device=answer.device, dtype=answer.dtype) 
                ans_out = self(answer, input_pos)
                logprobs = torch.cat([output["logits"][:,-1:,:], ans_out["logits"][:,:-1,:]], dim=1).log_softmax(dim=2)
                index = answer.unsqueeze(2)
                gather_probs = torch.gather(logprobs, -1, index).squeeze(2)
                ans_logit = gather_probs.sum()
                answers_logits.append(ans_logit)
            logits.append(torch.stack(answers_logits, dim=0))
            prompt_hidden_states.append(self._pool_embeddings(output["last_hidden_state"])[0])
        logits = torch.stack(logits, dim=0)
        prompt_hidden_states = torch.stack(prompt_hidden_states, dim=0)
        return {"logits": logits, "prompt_hidden_states": prompt_hidden_states}

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answers_ids: List[List[torch.Tensor]],
        labels: torch.Tensor,
        loss_fn: str = "cross_entropy",
    ):
        if loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")

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

    def _pool_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.embedding_pooling == "mean":
            return embeddings.mean(dim=1)
        elif self.embedding_pooling == "max":
            return embeddings.max(dim=1).values
        elif self.embedding_pooling == "last":
            return embeddings[:,-1,:]
        else:
            raise ValueError(f"Invalid embedding_pooling {self.embedding_pooling}")

        

class LoRALitGPTSequenceClassification(LoRALitGPT):

    def __init__(self, model_name_or_path: str, embedding_pooling: Literal["mean", "max", "last"], num_classes: int, **lora_kwargs):
        super().__init__(model_name_or_path, **lora_kwargs)
        self.embedding_pooling = embedding_pooling
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.config.n_embd, num_classes)

    def forward(
        self,
        prompt_ids: torch.Tensor, 
        prompt_mask: torch.Tensor, 
    ):
        embeddings = []
        for input_ids, attention_mask in zip(prompt_ids, prompt_mask):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            output = super().forward(input_ids)
            embeddings.append(self._pool_embeddings(output["last_hidden_state"])[0])
        embeddings = torch.stack(embeddings, dim=0)
        logits = self.classifier(embeddings)
        return {"logits": logits}
    
    def train_step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: str = "cross_entropy",
    ):
        if loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")
        
        outputs = self(prompt_ids, prompt_mask)["logits"]
        loss = nn.functional.cross_entropy(outputs, labels)
        return loss
    
    def predict_step(
        self,
        prompt_ids: torch.Tensor, 
        prompt_mask: torch.Tensor,
    ):
        return self(prompt_ids, prompt_mask)

    def _pool_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.embedding_pooling == "mean":
            return embeddings.mean(dim=1)
        elif self.embedding_pooling == "max":
            return embeddings.max(dim=1).values
        elif self.embedding_pooling == "last":
            return embeddings[:,-1,:]
        else:
            raise ValueError(f"Invalid embedding_pooling {self.embedding_pooling}")


        
    
        