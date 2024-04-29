
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Literal, Optional
import lightning as L
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
from .lit_tokenizer import LitGPTTokenizer
from litgpt import Config
from ...prompt import PrefixPrompt
from .lit_model import LitGPT
from litgpt.utils import load_checkpoint
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler

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


class LanguageModelLitGPTFullFT(L.LightningModule):

    def __init__(
        self,
        checkpoint_dir: str,
        preshots_template: str, 
        shots_template: str,
        postshots_template: str,
        shots_separator: str,
        answers_templates: List[str],
        data_load_fn: Callable,
        data_cache_dir: str,
        batch_size: int,
        loss_fn: Literal["cross_entropy"] = "cross_entropy",
        embedding_pooling: Literal["mean", "max", "last"] = "last"
    ):
        super().__init__()
        if not os.path.exists(checkpoint_dir):
            if not os.path.exists(os.path.join(os.getenv("LIT_CHECKPOINTS"), checkpoint_dir)):
                raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
            self.checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.config = Config.from_checkpoint(self.checkpoint_dir)
        self.tokenizer = LitGPTTokenizer(self.checkpoint_dir)
        self.checkpoint_path = self.checkpoint_dir / "lit_model.pth" # Set this to None if you don't want to load the weights from pretrained model
        self.prompt = PrefixPrompt(preshots_template, shots_template, postshots_template, shots_separator, answers_templates)
        self.data_load_fn = data_load_fn
        self.data_cache_dir = data_cache_dir
        self.batch_size = batch_size

        if loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")
        self.loss_fn = loss_fn

        self.embedding_pooling = embedding_pooling

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
    # Model initialization
    # --------------------------------------------------------------------------------------------
    def configure_model(self) -> None:
        self.pt_model = LitGPT(self.config)
        self.pt_model.set_kv_cache(batch_size=1)
        for param in self.pt_model.parameters():
            param.requires_grad = True

    def init_params(self) -> None:
        if self.checkpoint_path is not None:
            load_checkpoint(self.fabric, self.pt_model, self.checkpoint_path, strict=True)
    
    # --------------------------------------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------------------------------------
    def configure_optimizers(self) -> OptimizerLRScheduler:
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.AdamW(trainable_params[:1], lr=1e-4)
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        return
    
    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        return
    
    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        return super().lr_scheduler_step(scheduler, metric)

    # --------------------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        return self.pt_model(idx, input_pos, output_last_hidden_state)

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        return super().on_train_batch_start(batch, batch_idx)

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
            num_tokens = num_tokens + input_ids.shape[1]
            indices_counts = indices_counts + torch.bincount(index.view(-1), minlength=logprobs.shape[2])

        priors = indices_counts.float() / indices_counts.sum()
        naive_loss = -torch.sum(priors[priors > 0] * torch.log(priors[priors > 0]))
        self.fabric.log_dict({
            "train/cross_entropy_over_unigram": loss / num_tokens / naive_loss,
            "train/cross_entropy_per_token": loss / num_tokens,
        }, step=self.trainer.global_step)
        return {"loss": loss / num_tokens}
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()
    
    # --------------------------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------------------------
    def on_validation_model_eval(self) -> None:
        self.eval()

    def on_validation_epoch_start(self) -> None:
        self.cum_val_loss = 0
        self.cum_tokens = 0
        self.cum_indices_counts = 0

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

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
            num_tokens = num_tokens + input_ids.shape[1]
            indices_counts = indices_counts + torch.bincount(index.view(-1), minlength=logprobs.shape[2])

        self.cum_val_loss = self.cum_val_loss + loss
        self.cum_tokens = self.cum_tokens + num_tokens
        self.cum_indices_counts = self.cum_indices_counts + indices_counts
        return {"loss": loss / num_tokens}

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        self.avg_val_loss = self.cum_val_loss / self.cum_tokens
        priors = self.cum_indices_counts.float() / self.cum_indices_counts.sum()
        naive_loss = -torch.sum(priors[priors > 0] * torch.log(priors[priors > 0]))
        self.norm_cross_entropy = self.avg_val_loss / naive_loss
        self.fabric.log_dict({
            "val/cross_entropy_over_unigram": self.norm_cross_entropy,
            "val/cross_entropy_per_token": self.avg_val_loss,
        }, step=self.trainer.global_step)

    def on_validation_model_train(self) -> None:
        self.train()

    def on_fit_end(self) -> None:
        self.fabric.save(Path(self.trainer.checkpoint_dir) / "lit_model.pth", self.pt_model.state_dict())

        config_files = ["config.json", "generation_config.json", "model_config.yaml"]
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        for file_name in config_files + tokenizer_files:
            src_path = self.checkpoint_dir / file_name
            if src_path.exists():
                shutil.copy(src_path, self.trainer.checkpoint_dir)

    # --------------------------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------------------------
    def on_predict_start(self) -> None:
        return super().on_predict_start()

    def on_predict_epoch_start(self) -> None:
        self.eval()
    
    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_predict_batch_start(batch, batch_idx, dataloader_idx)
    
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
        return super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)
    
    def on_predict_epoch_end(self) -> None:
        return super().on_predict_epoch_end()
    
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

if __name__ == "__main__":
    from functools import partial
    from ...data.datasets import load_sst2
    model = LanguageModelLitGPTFullFT(
        checkpoint_dir = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        preshots_template = "",
        shots_template = "Review: \"{sentence}\"\nSentiment: {answer}",
        postshots_template = "Review: \"{sentence}\"\nSentiment:",
        shots_separator = "\n",
        answers_templates = ["Negative", "Positive"],
        data_load_fn = partial(
            load_sst2, 
            data_dir=f"experiments/sst2/.cache", 
            num_train_samples=100, 
            num_val_samples=100, 
            num_shots=2, 
            random_state=738
        ),
        data_cache_dir = f"experiments/sst2/n=100_rs=738/prefix_basic_sst2/lm_tinyllama_3T_bf16/.data_cache",
        batch_size = 32,
        loss_fn = "cross_entropy"
    )
    trainer = L.Trainer(accelerator="gpu", devices=1, precision="bf16-true", max_epochs=1)
    trainer.fit(model)