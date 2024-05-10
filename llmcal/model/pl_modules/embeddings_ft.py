from pathlib import Path
from typing import Any
from collections import defaultdict

import lightning as L
import torch

from collections import defaultdict
from torch import nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from ..metrics import norm_cross_entropy

class EmbeddingsFT(L.LightningModule):

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        max_ls: int = 40,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.input_features = input_features
        self.num_classes = num_classes
        self.max_ls = max_ls
        self.learning_rate = learning_rate

        self.classifier = nn.Linear(self.input_features, self.num_classes, bias=True)
        for param in self.parameters():
            param.requires_grad = True

    def configure_optimizers(self) -> OptimizerLRScheduler:
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.LBFGS(trainable_params, lr=self.learning_rate, max_iter=self.max_ls)
    
    def forward(self, embeddings) -> torch.Tensor:
        return self.classifier(embeddings)

    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        embeddings, labels = batch["prompt_hidden_states"].to(dtype=self.classifier.weight.dtype), batch["label"]

        train_ce = []
        norm_train_ce = []
        def closure():
            cal_logits = self(embeddings)
            loss = torch.nn.functional.cross_entropy(cal_logits, labels)
            optimizer.zero_grad()
            self.manual_backward(loss)
            train_ce.append(loss.item())
            norm_train_ce.append(norm_cross_entropy(cal_logits, labels).item())
            return loss
        
        loss = optimizer.step(closure)
        self.logger.log_metrics({
            "train/cross_entropy": train_ce[-1],
            "train/norm_cross_entropy": norm_train_ce[-1],
        }, step=self.global_step)
        return {"loss": loss}
        
    def on_validation_model_eval(self) -> None:
        self.eval()

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch["prompt_hidden_states"].to(dtype=self.calibrator.alpha.dtype), batch["label"]
        cal_logits = self(embeddings)
        loss = torch.nn.functional.cross_entropy(cal_logits, labels)
        self.logger.log_metrics({
            "val/cross_entropy": loss.item(),
            "val/norm_cross_entropy": norm_cross_entropy(cal_logits, labels).item(),
        }, step=self.global_step)
        self.log("val_loss", loss.item(), logger=False, on_epoch=True)
        return {"loss": loss}

    def on_validation_model_train(self) -> None:
        self.train()

    def on_fit_end(self) -> None:
        torch.save(self.classifier.state_dict(), Path(self.trainer.default_root_dir) / "classifier.pth")

    def on_predict_start(self) -> None:
        self.eval()

    def on_predict_epoch_start(self) -> None:
        self.predict_outputs = defaultdict(list)
    
    def predict_step(self, batch, batch_idx):
        embeddings, labels = batch["prompt_hidden_states"].to(dtype=self.classifier.alpha.dtype), batch["label"]
        cal_logits = self(embeddings)
        return {"idx": batch["idx"], "logits": cal_logits, "label": labels}

    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu())

    def on_predict_end(self) -> None:
        predict_outputs = {}
        for k, v in self.predict_outputs.items():
            predict_outputs[k] = torch.cat(v, dim=0)
        self.predict_outputs = predict_outputs
        self.train()
    