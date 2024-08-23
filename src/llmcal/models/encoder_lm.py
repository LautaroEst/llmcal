
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List, Literal, Optional
from datasets import Dataset

import torch
import lightning as L
from lightning.pytorch.trainer.states import RunningStage
import torch.nn.functional as F
    

class EncoderLanguageModel(L.LightningModule):

    def __init__(
        self,
        base_model,
        optimizer: Literal["adamw", "sgd"] = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self._optimizer_name = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

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

    def forward(self, batch, output_hidden_states=False):
        return self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch, output_hidden_states=False)
        logits = outputs["logits"]
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)
    
        for logger in self.loggers:
            logger.log_metrics(
                {"train/ce": loss.item()},
                step=self.global_step,
            )

        return loss
    
    def on_train_epoch_end(self):
        self.trainer.save_checkpoint(
            Path(self.trainer.default_root_dir) / f"last.ckpt"
        )

    def on_validation_epoch_start(self) -> None:
        self.val_cum_loss = 0.
        self.val_cum_samples = 0

    def validation_step(self, batch, batch_idx):
        outputs = self(batch, output_hidden_states=False)
        logits = outputs["logits"]
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)
        self.val_cum_loss += loss.item()
        self.val_cum_samples += len(labels)
        return {"loss": loss, "num_samples": len(labels)}

    def on_validation_epoch_end(self):
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            ce = self.val_cum_loss / self.val_cum_samples
            for logger in self.loggers:
                logger.log_metrics({
                    "val/ce": ce,
                }, step=self.global_step)

            if ce < self.best_val_loss:
                self.best_val_loss = ce
                self.trainer.save_checkpoint(
                    Path(self.trainer.default_root_dir) / "best.ckpt"
                )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["best_val_loss"] = self.best_val_loss

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.best_val_loss = checkpoint["best_val_loss"]

    def on_predict_start(self) -> None:
        self.eval()

    def on_predict_epoch_start(self) -> None:
        self.predict_outputs = defaultdict(list)

    def predict_step(self, batch, batch_idx):
        outputs = self(batch, output_hidden_states=True)
        logits = outputs["logits"]
        cls_emb = outputs["hidden_states"][-1][:,0,:]

        return {
            "idx": batch["idx"], 
            "logits": logits, 
            "cls_embedding": cls_emb, 
            "label": batch["labels"],
        }

    def on_predict_batch_end(self, outputs, batch, batch_idx) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu().float())

    def on_predict_end(self) -> None:
        predict_outputs = {}
        for k, v in self.predict_outputs.items():
            predict_outputs[k] = torch.cat(v, dim=0)
        self.predict_outputs = Dataset.from_dict(predict_outputs)
        
        
        
