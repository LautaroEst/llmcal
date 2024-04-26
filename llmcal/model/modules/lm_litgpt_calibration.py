
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
from torch.utils.data import TensorDataset
from .affine_calibration import AffineCalibrator
from ..losses import norm_cross_entropy


class LanguageModelLitGPTAffineCalibration(L.LightningModule):

    def __init__(
        self,
        data_cache_dir: str,
        alpha: Literal["matrix", "vector", "scalar", "none"] = "matrix",
        beta: bool = True,
        batch_size: int = 1,
    ):
        super().__init__()
        self.data_cache_dir = data_cache_dir
        self.alpha = alpha
        self.beta = beta

        if os.path.exists(self.data_cache_dir):
            data = torch.load(os.path.join(self.data_cache_dir, "predict_train.pt"))
            self.num_classes = data["logits"].shape[1]
        else:
            raise ValueError(f"Model with no adaptation needs to be run first.")
        
        self.automatic_optimization = False
        self.batch_size = batch_size

    # --------------------------------------------------------------------------------------------
    # Data methods
    # --------------------------------------------------------------------------------------------
    def prepare_data(self):
        super().prepare_data()

    def setup(self, stage):
        datadict = {}
        for split in ["train", "val", "test"]:
            data = torch.load(os.path.join(self.data_cache_dir, f"predict_{split}.pt"))
            datadict[split] = TensorDataset(data["idx"], data["logits"], data["label"])

        if stage == "fit":
            self.train_data = datadict["train"]
            self.val_data = datadict["val"]
        elif stage == "test":
            self.test_data = datadict["test"]
        elif stage == "predict":
            self.predict_data = {"val": datadict["val"], "test": datadict["test"]}
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=len(self.train_data), shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=len(self.val_data), shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=len(self.test_data), shuffle=False)
    
    def predict_dataloader(self):
        return {
            split: DataLoader(self.predict_data[split], batch_size=self.batch_size, shuffle=False) \
            for split in ["val", "test"]
        }
    
    # --------------------------------------------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------------------------------------------
    def configure_model(self) -> None:
        with self.fabric.init_module():
            self.pt_model = AffineCalibrator(self.num_classes, self.alpha, self.beta)
        for param in self.pt_model.parameters():
            param.requires_grad = True
        self.pt_model.init_params(self.fabric)
    
    # --------------------------------------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------------------------------------
    def configure_optimizers(self) -> OptimizerLRScheduler:
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.LBFGS(trainable_params, lr=1.0, max_iter=40)
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        return
    
    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        return
    
    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        return super().lr_scheduler_step(scheduler, metric)

    # --------------------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------------------
    def forward(self, logits) -> torch.Tensor:
        return self.pt_model(logits)

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        idx, logits, labels = batch

        global step_count
        step_count = int(self.fabric.global_step)

        def closure():
            global step_count
            optimizer.zero_grad()
            cal_logits = self(logits)
            loss = torch.nn.functional.cross_entropy(cal_logits, labels)
            self.fabric.log_dict({
                "train/cross_entropy": loss.item(),
                "train/norm_cross_entropy": norm_cross_entropy(cal_logits, labels).item(),
            }, step=step_count)
            self.fabric.backward(loss)
            step_count += 1
            return {"loss": loss}
        
        outputs = optimizer.step(closure)
        self.fabric.global_step += step_count
        return outputs
    
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
        return super().on_validation_epoch_start()

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx):
        idx, logits, labels = batch
        cal_logits = self(logits)
        loss = torch.nn.functional.cross_entropy(cal_logits, labels)
        self.fabric.log_dict({
            "val/cross_entropy": loss.item(),
            "val/norm_cross_entropy": norm_cross_entropy(cal_logits, labels).item(),
        }, step=self.fabric.global_step)
        return {"loss": loss}

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()

    def on_validation_model_train(self) -> None:
        self.train()

    def on_fit_end(self) -> None:
        return super().on_fit_end()
    
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