from pathlib import Path
from typing import Any, Literal
from collections import defaultdict

import lightning as L
import torch
from torch import nn
from lightning.pytorch.trainer.states import RunningStage

from datasets import Dataset

class OutputLayer(nn.Module):
    
    def __init__(
        self, 
        num_classes: int, 
        embedding_dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


class EmbeddingsFT(L.LightningModule):

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        pooling: Literal["mean", "max", "last"] = "last",
        max_ls: int = 40,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.max_ls = max_ls
        self.learning_rate = learning_rate

        self.output_layer = OutputLayer(self.num_classes, self.embedding_dim)
        for param in self.parameters():
            param.requires_grad = True

        self.super_global_step = 0
        self.best_val_loss = float("inf")

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.LBFGS(trainable_params, lr=self.learning_rate, max_iter=self.max_ls)
    
    def forward(self, embeddings) -> torch.Tensor:
        return self.output_layer(embeddings)
    
    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        embeddings, labels = batch[f"{self.pooling}_embeddings"].float(), batch["label"].long()

        def closure():
            cal_logits = self(embeddings)
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
        embeddings, labels = batch[f"{self.pooling}_embeddings"].float(), batch["label"].long()
        cal_logits = self(embeddings)
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
        embeddings, labels = batch[f"{self.pooling}_embeddings"].float(), batch["label"].long()
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
        self.predict_outputs = Dataset.from_dict(predict_outputs)
    