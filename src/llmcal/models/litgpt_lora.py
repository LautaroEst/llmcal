
from collections import defaultdict
import os
import shutil
from pathlib import Path
from typing import List, Literal, Optional

import torch
import lightning as L
from lightning import lazy_load
from litgpt.lora import Config as LoraConfig, GPT as LitGPTLoRA, mark_only_lora_as_trainable, LoRALinear, merge_lora_weights, lora_filter

    
def init_lora_linear_modules(module):
    if isinstance(module, LoRALinear):
        module.reset_parameters()
    else:
        for child in module.children():
            init_lora_linear_modules(child)



class LanguageModelLitGPTLoRA(L.LightningModule):

    def __init__(
        self,
        checkpoint_dir: str,
        embedding_pooling: Literal["mean", "max", "last"],
        use_lora_checkpoint: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_query: bool = True,
        lora_key: bool = False,
        lora_value: bool = True,
        lora_projection: bool = False,
        lora_mlp: bool = False,
        lora_head: bool = False,
        optimizer: Literal["adamw", "sgd"] = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        # Init config and tokenizer
        if not os.path.exists(checkpoint_dir):
            if not os.path.exists(os.path.join(os.getenv("LIT_CHECKPOINTS"), checkpoint_dir)):
                raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
            self.checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.config = LoraConfig.from_checkpoint(
            self.checkpoint_dir,
            lora_r = lora_r,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            lora_query = lora_query,
            lora_key = lora_key,
            lora_value = lora_value,
            lora_projection = lora_projection,
            lora_mlp = lora_mlp,
            lora_head = lora_head,
        )
        self.checkpoint_path = self.checkpoint_dir / "lit_model.pth" # Set this to None if you don't want to load the weights from pretrained model
        if use_lora_checkpoint:
            self.checkpoint_path_lora = self.checkpoint_dir / "lit_model.pth.lora" 
        else:
            self.checkpoint_path_lora = None

        # Init model
        self.gpt = LitGPTLoRA(self.config)
        self.gpt.set_kv_cache(batch_size=1)
        mark_only_lora_as_trainable(self.gpt)

        if self.checkpoint_path is not None:
            checkpoint = lazy_load(self.checkpoint_path)
            checkpoint = {"gpt." + k: v for k, v in checkpoint.items()}
            self.load_state_dict(checkpoint, strict=False)
            if self.checkpoint_path_lora is not None:
                checkpoint = lazy_load(self.checkpoint_path_lora)
                checkpoint = {"gpt." + k: v for k, v in checkpoint.items()}
                self.load_state_dict(checkpoint, strict=False)
            else:
                init_lora_linear_modules(self.gpt)

        self.embedding_pooling = embedding_pooling
        
        # Training args
        self._optimizer_name = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if self._optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        elif self._optimizer_name == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        else:
            raise ValueError(f"Invalid optimizer {self._optimizer_name}")
        return optimizer

    def on_fit_end(self) -> None:
        torch.save({k: v for k, v in self.gpt.state_dict().items() if lora_filter(k,v)}, Path(self.trainer.default_root_dir) / "model" / "lit_model.pth.lora")
        if self.checkpoint_path is not None:
            os.symlink(self.checkpoint_path, os.path.join(self.trainer.default_root_dir,"model/lit_model.pth")) # symlink to the "lit_model.pth" file
        else:
            torch.save({k: v for k, v in self.gpt.state_dict().items() if not lora_filter(k,v)}, Path(self.trainer.default_root_dir) / "model" / "lit_model.pth")

        config_files = ["config.json", "generation_config.json", "model_config.yaml"]
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        for file_name in config_files + tokenizer_files:
            src_path = self.checkpoint_dir / file_name
            if src_path.exists():
                shutil.copy(src_path, os.path.join(self.trainer.default_root_dir,"model"))
        
        # Merge lora weights for predictions after saving them
        merge_lora_weights(self.gpt)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        return self.gpt(idx, input_pos, output_last_hidden_state)
    
    def on_train_start(self) -> None:
        self.last_global_step = 0
        self.cum_loss = 0.
        self.cum_num_tokens = 0

    def train_step(self, batch, batch_idx):
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        
        loss = 0
        num_tokens = 0
        for input_ids, attention_mask in zip(prompt_ids, prompt_mask):
            logprobs = self(input_ids[attention_mask == 1].unsqueeze(0))["logits"][:,:-1,:].log_softmax(dim=2)
            index = input_ids[:,1:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + index.size(1)

        return {"loss": loss / num_tokens, "cum_loss": loss, "num_tokens": num_tokens}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.last_global_step == self.global_step:
            self.cum_loss += outputs["cum_loss"]
            self.cum_num_tokens += outputs["num_tokens"]
            return
        
        self.logger.log_metrics(
            {"train/ce_per_token": self.cum_loss / self.cum_num_tokens},
            step=self.last_global_step,
        )
        self.last_global_step = self.global_step
        self.cum_loss = 0.
        self.cum_num_tokens = 0

    def on_validation_epoch_start(self) -> None:
        self.val_epoch_logits = []
        self.val_epoch_labels = []
        self.eval()

    def validation_step(self, batch, batch_idx):
        prompt_ids: torch.Tensor = batch["prompt_ids"]
        prompt_mask: torch.Tensor = batch["prompt_mask"]
        answers_ids: List[List[torch.Tensor]] = batch["answers_ids"]

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
        self.val_epoch_logits.append(logits)
        self.val_epoch_labels.append(batch["label"])
        
    def on_validation_epoch_end(self):
        logits = torch.stack(self.val_epoch_logits, dim=0)
        labels = torch.stack(self.val_epoch_labels, dim=0)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        priors = torch.bincount(labels, minlength=logits.size(1)).float() / labels.size(0)
        ent = -torch.mean(torch.log(priors[labels]))
        er = 1 - torch.mean((logits.argmax(dim=1) == labels).float())
        prior_er = 1 - torch.mean((priors[labels].argmax(dim=1) == labels).float())
        self.logger.log_metrics({
            "val/cross_entropy": ce.item(),
            "val/norm_cross_entropy": ce.item() / ent.item(), 
            "val/error_rate": er.item(),
            "val/norm_error_rate": er.item() / prior_er.item(),
        }, step=self.global_step)
        self.train()

    def on_predict_start(self) -> None:
        self.eval()

    def on_predict_epoch_start(self) -> None:
        self.predict_outputs = defaultdict(list)

    def predict_step(self, batch, batch_idx):
        prompt_ids: torch.Tensor = batch["prompt_ids"]
        prompt_mask: torch.Tensor = batch["prompt_mask"]
        answers_ids: List[List[torch.Tensor]] = batch["answers_ids"]

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

        return {"idx": batch["idx"], "logits": logits, "prompt_hidden_states": prompt_hidden_states, "label": batch["label"]}

    def on_predict_batch_end(self, outputs, batch, batch_idx) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu())

    def on_predict_end(self) -> None:
        predict_outputs = {}
        for k, v in self.predict_outputs.items():
            predict_outputs[k] = torch.cat(v, dim=0)
        self.predict_outputs = predict_outputs
        self.train()
    
    def _pool_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.embedding_pooling == "mean":
            return embeddings.mean(dim=1)
        elif self.embedding_pooling == "max":
            return embeddings.max(dim=1).values
        elif self.embedding_pooling == "last":
            return embeddings[:,-1,:]
        else:
            raise ValueError(f"Invalid embedding_pooling {self.embedding_pooling}")

    

        
        
        
        
