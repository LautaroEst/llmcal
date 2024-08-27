from pathlib import Path
from typing import Any, Literal
from collections import defaultdict

import lightning as L
import torch
from torch import nn
from lightning.pytorch.trainer.states import RunningStage

from datasets import Dataset

class MahalanobisCalibrator(nn.Module):
    """
    Affine calibration block. It is a linear block that performs an affine transformation
    of the input feature vector ino order to output the calibrated logits.

    Parameters
    ----------
    num_features : int
        Number of input features of the calibrator.
    num_classes : int
        Number of output classes of the calibrator.
    alpha : {"vector", "scalar", "matrix", "none"}, optional
        Type of affine transformation, by default "vector"
    beta : bool, optional
        Whether to use a beta term, by default True
    """    
    def __init__(
        self, 
        num_classes: int, 
        # alpha: Literal["matrix", "vector", "scalar", "none"] = "matrix",
        # beta: bool = True,
        eps: float = 1e-6,
        random_state: int = 42,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.additional_arguments = {
            "eps": eps,
            "random_state": random_state
        }

        # Parameters
        self.means = nn.Parameter(torch.zeros(num_classes,num_classes), requires_grad=True)
        # self.covariances = nn.Parameter(torch.zeros(num_classes,num_classes,num_classes), requires_grad=False)
    
    def init_parameters(self, train_data):
        logits, labels = train_data["logits"].float(), train_data["label"].long()
        logits = torch.log_softmax(logits, dim=1)

        device = self.means.device
        for c in range(self.num_classes):
            features_c = logits[labels == c].to(device)
            self.means.data[c] = torch.mean(features_c, dim=0)

    def forward(self, features):

        device = self.means.device
        covariances = torch.zeros(self.num_classes, self.num_classes, self.num_classes, device=device)
        for c in range(self.num_classes):
            features_c = self.train_features[self.train_labels == c].to(device)
            centered_features = features_c - self.means[c]
            covariances[c] = torch.matmul(centered_features.T, centered_features) / (features_c.shape[0] - 1) + self.eps * torch.eye(self.num_features, device=features_c.device)

        inv_sigma = torch.cholesky_inverse(covariances)
        features_centered = features.unsqueeze(1) - self.means
        inv_sigma_features = torch.matmul(inv_sigma.unsqueeze(0), features_centered.unsqueeze(3)).squeeze(3)
        mahalanobis = torch.sum(features_centered * inv_sigma_features, dim=2)
        cal_logits = torch.log_softmax(-mahalanobis)
        return cal_logits


class MahalanobisCalibration(L.LightningModule):

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-6,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.num_classes = num_classes
        self.eps = eps
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.calibrator = MahalanobisCalibrator(self.num_classes, self.eps, self.random_state)
        
        self.super_global_step = 0
        self.best_val_loss = float("inf")

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if self._optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        elif self._optimizer_name == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=self._learning_rate, weight_decay=self._weight_decay)
        else:
            raise ValueError(f"Invalid optimizer {self._optimizer_name}")
        return optimizer
    
    def forward(self, logits) -> torch.Tensor:
        return self.calibrator(logits)
    
    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        logits, labels = batch["logits"].float(), batch["label"].long()
        logits = torch.log_softmax(logits, dim=1)

        def closure():
            cal_logits = self(logits)
            loss = torch.nn.functional.cross_entropy(cal_logits, labels)
            optimizer.zero_grad()
            self.manual_backward(loss)
            for logger in self.loggers:
                logger.log_metrics({
                    "train/cross_entropy": loss.item(),
                }, step=self.super_global_step)
            self.super_global_step += 1
            return loss
        
        loss = optimizer.step(closure)
        return {"loss": loss}

    def on_train_epoch_end(self):
        self.trainer.save_checkpoint(
            Path(self.trainer.default_root_dir) / f"last.ckpt"
        )

    def validation_step(self, batch, batch_idx):
        logits, labels = batch["logits"].float(), batch["label"].long()
        logits = torch.log_softmax(logits, dim=1)
        cal_logits = self(logits)
        loss = torch.nn.functional.cross_entropy(cal_logits, labels)
        
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            for logger in self.loggers:
                logger.log_metrics({
                    "val/cross_entropy": loss.item(),
                }, step=self.super_global_step)
            if loss.item() < self.best_val_loss:
                self.best_val_loss = loss.item()
                self.trainer.save_checkpoint(
                    Path(self.trainer.default_root_dir) / "best.ckpt"
                )
        return {"loss": loss}

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        checkpoint["super_global_step"] = self.super_global_step
        checkpoint["best_val_loss"] = self.best_val_loss

    def on_load_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        self.super_global_step = checkpoint["super_global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

    def on_predict_start(self) -> None:
        self.eval()

    def on_predict_epoch_start(self) -> None:
        self.predict_outputs = defaultdict(list)
    
    def predict_step(self, batch, batch_idx):
        logits, labels = batch["logits"].float(), batch["label"].long()
        logits = torch.log_softmax(logits, dim=1)
        cal_logits = self(logits)
        return {"idx": batch["idx"], "logits": cal_logits, "label": labels}

    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu())

    def on_predict_end(self) -> None:
        predict_outputs = {}
        for k, v in self.predict_outputs.items():
            predict_outputs[k] = torch.cat(v, dim=0)
        self.predict_outputs = predict_outputs
        self.predict_outputs = Dataset.from_dict(predict_outputs)
    