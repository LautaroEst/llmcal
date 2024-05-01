
from collections import defaultdict
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Literal, Optional
import lightning as L
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
from ..base_classes import LitGPTTokenizer, AffineCalibrator
from litgpt import Config
from ..prompt import PrefixPrompt
from litgpt.utils import load_checkpoint
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.utils.data import TensorDataset
from ..metrics import norm_cross_entropy


class LanguageModelLitGPTAffineCalibration(L.LightningModule):

    def __init__(
        self,
        data_cache_dir: str,
        alpha: Literal["matrix", "vector", "scalar", "none"] = "matrix",
        beta: bool = True,
        batch_size: int = 1,
        max_ls: int = 40,
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
        
        self.batch_size = batch_size
        self.max_ls = max_ls

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
        self.pt_model = AffineCalibrator(self.num_classes, self.alpha, self.beta)
        for param in self.pt_model.parameters():
            param.requires_grad = True

    def init_params(self) -> None:
        self.pt_model.alpha.data.fill_(1.)
        self.pt_model.beta.data.fill_(0.)
    
    # --------------------------------------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------------------------------------
    def configure_optimizers(self) -> OptimizerLRScheduler:
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.LBFGS(trainable_params, lr=0.001, max_iter=self.max_ls)
    
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

        idx, logits, labels = batch

        self.optimizers().zero_grad()

        cal_logits = self(logits)
        loss = torch.nn.functional.cross_entropy(cal_logits, labels)
        self.fabric.log_dict({
            "train/cross_entropy": loss.item(),
            "train/norm_cross_entropy": norm_cross_entropy(cal_logits, labels).item(),
        }, step=self.trainer.global_step)
        
        return {"loss": loss}
    
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
        self.avg_val_loss = loss
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
        self.predict_outputs = defaultdict(list)
    
    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_predict_batch_start(batch, batch_idx, dataloader_idx)
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        idx, logits, labels = batch
        cal_logits = self(logits)
        self._curr_out = {"idx": idx, "logits": cal_logits, "labels": labels}
        return self._curr_out

    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu())
    
    def on_predict_epoch_end(self) -> None:
        for k, v in self.predict_outputs.items():
            self.predict_outputs[k] = torch.cat(v, dim=0)
    
    def on_predict_end(self) -> None:
        self.train()