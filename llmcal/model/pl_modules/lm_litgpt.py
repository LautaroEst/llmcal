import os
from pathlib import Path
import shutil
from typing import Any, List, Literal, Optional
from collections import defaultdict

import lightning as L
import torch
from litgpt.lora import Config as LoraConfig, mark_only_lora_as_trainable, lora_filter, merge_lora_weights
from litgpt import Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.trainer.states import RunningStage

from .utils import init_lora_linear_modules
from ..base_classes import LitGPTLoRA, LitGPT

from collections import defaultdict


class _LanguageModelLitGPT(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.last_global_step = 0

    # --------------------------------------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------------------------------------
    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if self._optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        elif self._optimizer_name == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        else:
            raise ValueError(f"Invalid optimizer {self._optimizer_name}")
        return optimizer

    # --------------------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        return self.gpt(idx, input_pos, output_last_hidden_state)

    def on_train_epoch_start(self) -> None:
        self.cum_indices_counts = 0
        self.cum_train_ce = 0.
        self.cum_num_tokens = 0
    
    def training_step(self, batch, batch_idx):
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        answers_ids = batch["answers_ids"]
        labels = batch["label"]

        loss = 0
        num_tokens = 0
        indices_counts = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = torch.cat([
                input_ids[attention_mask == 1].unsqueeze(0),
                answers[label.item()].unsqueeze(0)
            ], dim=1)
            logprobs = self(input_ids)["logits"][:,:-1,:].log_softmax(dim=2)
            index = input_ids[:,1:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + index.shape[1]
            indices_counts = indices_counts + torch.bincount(index.view(-1), minlength=logprobs.shape[2])

        self.cum_train_ce += loss.item()
        self.cum_num_tokens += num_tokens
        self.cum_indices_counts = self.cum_indices_counts + indices_counts
        return {"loss": loss / num_tokens, "num_tokens": num_tokens, "indices_counts": indices_counts}
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.current_train_batch_idx = batch_idx
        if self.last_global_step == self.global_step:
            return

        priors = self.cum_indices_counts.float() / self.cum_indices_counts.sum()
        naive_ce_per_token = -torch.sum(priors[priors > 0] * torch.log(priors[priors > 0]))

        self.logger.log_metrics({
            "train/norm_ce_per_token": self.cum_train_ce / self.cum_num_tokens / naive_ce_per_token,
            "train/ce_per_token": self.cum_train_ce / self.cum_num_tokens,
        }, step=self.global_step)

        self.cum_indices_counts = 0
        self.cum_train_ce = 0.
        self.cum_num_tokens = 0
        self.last_global_step = self.global_step

    # --------------------------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------------------------
    def on_validation_model_eval(self) -> None:
        self.eval()

    def on_validation_epoch_start(self) -> None:
        self.cum_indices_counts = 0
        self.cum_val_ce = 0.
        self.cum_val_tokens = 0

    def validation_step(self, batch, batch_idx):
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        answers_ids = batch["answers_ids"]
        labels = batch["label"]

        loss = 0
        num_tokens = 0
        indices_counts = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = torch.cat([
                input_ids[attention_mask == 1].unsqueeze(0),
                answers[label.item()].unsqueeze(0)
            ], dim=1)
            logprobs = self(input_ids)["logits"][:,:-1,:].log_softmax(dim=2)
            index = input_ids[:,1:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + index.shape[1]
            indices_counts = indices_counts + torch.bincount(index.view(-1), minlength=logprobs.shape[2])

        self.cum_indices_counts = self.cum_indices_counts + indices_counts
        self.cum_val_ce += loss.item()
        self.cum_val_tokens += num_tokens
        return {"val/ce_per_token": loss / num_tokens}

    def on_validation_epoch_end(self) -> None:
        val_ce_per_token = self.cum_val_ce / self.cum_val_tokens
        priors = self.cum_indices_counts.float() / self.cum_indices_counts.sum()
        naive_ce_per_token = -torch.sum(priors[priors > 0] * torch.log(priors[priors > 0])).item()
        val_norm_ce_per_token = val_ce_per_token / naive_ce_per_token
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            self.logger.log_metrics({
                "val/norm_ce_per_token": val_norm_ce_per_token,
                "val/ce_per_token": val_ce_per_token,
            }, step=self.global_step)
        self.log("val_loss", val_ce_per_token, logger=False, on_epoch=True, batch_size=1)

    def on_validation_model_train(self) -> None:
        self.train()    

    # --------------------------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------------------------
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

    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
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


class LanguageModelLitGPTFullFT(_LanguageModelLitGPT):
    
    def __init__(
        self,
        checkpoint_dir: str,
        embedding_pooling: Literal["mean", "max", "last"],
        loss_fn: Literal["cross_entropy"] = "cross_entropy",
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
        self.config = Config.from_checkpoint(self.checkpoint_dir)
        self.checkpoint_path = self.checkpoint_dir / "lit_model.pth" # Set this to None if you don't want to load the weights from pretrained model

        # Init model
        self.gpt = LitGPT(self.config)
        self.gpt.set_kv_cache(batch_size=1)
        for param in self.parameters():
            param.requires_grad = True

        if self.checkpoint_path is not None:
            checkpoint = lazy_load(self.checkpoint_path)
            checkpoint = {"gpt." + k: v for k, v in checkpoint.items()}
            self.load_state_dict(checkpoint, strict=False)

        self.embedding_pooling = embedding_pooling
        
        # Training args
        if loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")
        self.loss_fn = loss_fn
        self._optimizer_name = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def on_fit_end(self) -> None:
        torch.save(self.gpt.state_dict(), Path(self.trainer.default_root_dir) / "lit_model.pth")

        config_files = ["config.json", "generation_config.json", "model_config.yaml"]
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        for file_name in config_files + tokenizer_files:
            src_path = self.checkpoint_dir / file_name
            if src_path.exists():
                shutil.copy(src_path, self.trainer.default_root_dir)


class LanguageModelLitGPTLoRA(_LanguageModelLitGPT):

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
        loss_fn: Literal["cross_entropy"] = "cross_entropy",
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
        lora_kwargs = dict(
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
        self.config = LoraConfig.from_checkpoint(self.checkpoint_dir, **lora_kwargs)
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
        if loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")
        self.loss_fn = loss_fn
        self._optimizer_name = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

        # Metrics
        self.last_global_step = 0

    def on_fit_end(self) -> None:
        torch.save({k: v for k, v in self.gpt.state_dict().items() if lora_filter(k,v)}, Path(self.trainer.default_root_dir) / "lit_model.pth.lora")
        if self.checkpoint_path is not None:
            os.symlink(self.checkpoint_path, os.path.join(self.trainer.default_root_dir,"lit_model.pth")) # symlink to the "lit_model.pth" file
        else:
            torch.save({k: v for k, v in self.gpt.state_dict().items() if not lora_filter(k,v)}, Path(self.trainer.default_root_dir) / "lit_model.pth")

        config_files = ["config.json", "generation_config.json", "model_config.yaml"]
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        for file_name in config_files + tokenizer_files:
            src_path = self.checkpoint_dir / file_name
            if src_path.exists():
                shutil.copy(src_path, self.trainer.default_root_dir)
        
        # Merge lora weights for predictions after saving them
        merge_lora_weights(self.gpt)


class LanguageModelLitGPTNoAdaptation(_LanguageModelLitGPT):
    def __init__(
        self,
        checkpoint_dir: str,
        embedding_pooling: Literal["mean", "max", "last"],
    ):
        super().__init__()

        # Init config and tokenizer
        if not os.path.exists(checkpoint_dir):
            if not os.path.exists(os.path.join(os.getenv("LIT_CHECKPOINTS"), checkpoint_dir)):
                raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
            self.checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.config = Config.from_checkpoint(self.checkpoint_dir)
        self.checkpoint_path = self.checkpoint_dir / "lit_model.pth" # Set this to None if you don't want to load the weights from pretrained model

        # Init model
        self.gpt = LitGPT(self.config)
        self.gpt.set_kv_cache(batch_size=1)
        for param in self.parameters():
            param.requires_grad = True

        if self.checkpoint_path is not None:
            checkpoint = lazy_load(self.checkpoint_path)
            checkpoint = {"gpt." + k: v for k, v in checkpoint.items()}
            self.load_state_dict(checkpoint, strict=False)

        self.embedding_pooling = embedding_pooling
        
    def configure_optimizers(self):
        return

    def on_train_batch_start(self, batch, batch_idx):
        return

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        return

    def on_validation_model_eval(self) -> None:
        return

    def on_validation_epoch_start(self) -> None:
        return

    def on_validation_batch_start(self, batch: Any, batch_idx: int) -> None:
        return

    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        return

    def on_validation_epoch_end(self) -> None:
        return

    def on_validation_model_train(self) -> None:
        return

    def on_fit_end(self) -> None:
        os.symlink(self.checkpoint_path, os.path.join(self.trainer.default_root_dir,"lit_model.pth")) # symlink to the "lit_model.pth" file

        config_files = ["config.json", "generation_config.json", "model_config.yaml"]
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        for file_name in config_files + tokenizer_files:
            src_path = self.checkpoint_dir / file_name
            if src_path.exists():
                shutil.copy(src_path, self.trainer.default_root_dir)

    def on_predict_start(self) -> None:
        maybe_computed_outputs = list(self.checkpoint_dir.glob("*--predict.pt"))
        if not maybe_computed_outputs:
            self.eval()
            return
        
        for path in maybe_computed_outputs:
            os.symlink(path, os.path.join(self.trainer.default_root_dir, path.name))

        with open(os.path.join(self.trainer.default_root_dir, "done.txt"), "w") as f:
            f.write(self.trainer.default_root_dir)

        raise KeyboardInterrupt("Predictions already computed, exiting...")





