import os
from pathlib import Path
import shutil
from typing import Any, Callable, List, Literal, Optional
from collections import defaultdict

import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from litgpt import Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from torchmetrics.aggregation import MeanMetric

from .utils import DynamicPaddingCollator
from ..base_classes import LitGPT, LitGPTTokenizer
from ..prompt import PrefixPrompt


class LanguageModelLitGPTFullFT(L.LightningModule):

    def __init__(
        self,
        checkpoint_dir: str,
        embedding_pooling: Literal["mean", "max", "last"],
        preshots_template: str, 
        shots_template: str,
        postshots_template: str,
        shots_separator: str,
        answers_templates: List[str],
        data_load_fn: Callable,
        data_cache_dir: str,
        batch_size: int,
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
        self.tokenizer = LitGPTTokenizer(self.checkpoint_dir)
        self.checkpoint_path = self.checkpoint_dir / "lit_model.pth" # Set this to None if you don't want to load the weights from pretrained model
        
        # Init model
        self.gpt = LitGPT(self.config)
        self.gpt.set_kv_cache(batch_size=1)
        for param in self.gpt.parameters():
            param.requires_grad = True
        checkpoint = lazy_load(self.checkpoint_path)
        checkpoint = {"gpt." + k: v for k, v in checkpoint.items()}
        self.load_state_dict(checkpoint)
        self.embedding_pooling = embedding_pooling
        
        # Init prompt
        self.prompt = PrefixPrompt(preshots_template, shots_template, postshots_template, shots_separator, answers_templates)
        self.data_load_fn = data_load_fn
        self.data_cache_dir = data_cache_dir
        self.batch_size = batch_size

        # Training args
        if loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")
        self.loss_fn = loss_fn
        self._optimizer_name = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

        # Metrics
        self.val_ce_per_token = MeanMetric()
        self.train_ce_per_token = MeanMetric()
        self.last_global_step = 0

    # --------------------------------------------------------------------------------------------
    # Data methods
    # --------------------------------------------------------------------------------------------
    def prepare_data(self):

        if os.path.exists(self.data_cache_dir):
            return
        
        # Create the cache directory
        os.makedirs(self.data_cache_dir, exist_ok=True)

        # Download the dataset
        datadict, shots = self.data_load_fn()

        # Fill the prompt and tokenize
        self.prompt.fit(shots)

        def transform(sample):
            prompt = self.prompt.transform(**sample)
            prompt_ids = self.tokenizer([prompt["prompt"]])["input_ids"][0,:]
            answers_ids = [self.tokenizer([ans])["input_ids"][0,1:] for ans in prompt["answers"]]
            return {"idx": sample["idx"], "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]}

        # Process the dataset
        datadict["train"] = datadict["train"].map(transform)
        datadict["validation"] = datadict["validation"].map(transform)
        datadict["test"] = datadict["test"].map(transform)

        # Save to disk
        datadict.save_to_disk(self.data_cache_dir)

    def setup(self, stage):
        datadict = load_from_disk(self.data_cache_dir)
        if stage == "fit":
            self.train_data = datadict["train"].with_format("torch")
            self.val_data = datadict["validation"].with_format("torch")
        elif stage == "test":
            self.test_data = datadict["test"].with_format("torch")
        elif stage == "predict":
            self.predict_data = {"val": datadict["validation"].with_format("torch"), "test": datadict["test"].with_format("torch")}
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collator)
    
    def val_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collator)
    
    def test_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=collator)
    
    def predict_dataloader(self):
        collator = DynamicPaddingCollator(self.tokenizer.pad_token_id, self.config.block_size)
        return {
            split: DataLoader(self.predict_data[split], batch_size=self.batch_size, shuffle=False, collate_fn=collator) \
            for split in ["val", "test"]
        }
    
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

        self.train_ce_per_token.update(loss / num_tokens, num_tokens)
        self.cum_indices_counts = self.cum_indices_counts + indices_counts
        return {"loss": loss / num_tokens, "num_tokens": num_tokens, "indices_counts": indices_counts}
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.last_global_step == self.global_step:
            return

        ce_per_token = self.train_ce_per_token.compute()

        priors = self.cum_indices_counts.float() / self.cum_indices_counts.sum()
        naive_ce_per_token = -torch.sum(priors[priors > 0] * torch.log(priors[priors > 0]))

        self.logger.log_metrics({
            "train/norm_ce_per_token": ce_per_token / naive_ce_per_token,
            "train/ce_per_token": ce_per_token,
        }, step=self.global_step)

        self.cum_indices_counts = 0
        self.train_ce_per_token.reset()
        self.last_global_step = self.global_step

    # --------------------------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------------------------
    def on_validation_model_eval(self) -> None:
        self.eval()

    def on_validation_epoch_start(self) -> None:
        self.cum_indices_counts = 0
        self.val_ce_per_token.reset()

    def on_validation_batch_start(self, batch: Any, batch_idx: int) -> None:
        return super().on_validation_batch_start(batch, batch_idx)

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
        self.val_ce_per_token.update(loss / num_tokens, num_tokens)
        val_ce_per_token = self.val_ce_per_token.compute()
        return {"val/ce_per_token": val_ce_per_token}

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        return super().on_validation_batch_end(outputs, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        val_ce_per_token = self.val_ce_per_token.compute()
        priors = self.cum_indices_counts.float() / self.cum_indices_counts.sum()
        naive_ce_per_token = -torch.sum(priors[priors > 0] * torch.log(priors[priors > 0]))
        val_norm_ce_per_token = val_ce_per_token / naive_ce_per_token
        self.logger.log_metrics({
            "val/norm_ce_per_token": val_norm_ce_per_token,
            "val/ce_per_token": val_ce_per_token,
        }, step=self.global_step)
        self.log("val_loss", val_ce_per_token, logger=False, on_epoch=True, batch_size=1)

    def on_validation_model_train(self) -> None:
        self.train()

    def on_fit_end(self) -> None:
        torch.save(self.gpt.state_dict(), Path(self.trainer.default_root_dir) / "lit_model.pth")

        config_files = ["config.json", "generation_config.json", "model_config.yaml"]
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        for file_name in config_files + tokenizer_files:
            src_path = self.checkpoint_dir / file_name
            if src_path.exists():
                shutil.copy(src_path, self.trainer.default_root_dir)

    # --------------------------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------------------------
    def on_predict_epoch_start(self) -> None:
        self.eval()
        self.predict_outputs = defaultdict(list)
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
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
                ans_logit = gather_probs.mean() # changed from .sum()
                answers_logits.append(ans_logit)
            logits.append(torch.stack(answers_logits, dim=0))
            prompt_hidden_states.append(self._pool_embeddings(output["last_hidden_state"])[0])
        logits = torch.stack(logits, dim=0)
        prompt_hidden_states = torch.stack(prompt_hidden_states, dim=0)

        return {"idx": batch["idx"], "logits": logits, "prompt_hidden_states": prompt_hidden_states, "label": batch["label"]}

    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu())
    
    def on_predict_epoch_end(self) -> None:
        for k, v in self.predict_outputs.items():
            self.predict_outputs[k] = torch.cat(v, dim=0)
    
    def on_predict_end(self) -> None:
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
