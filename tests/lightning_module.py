
import os
from pathlib import Path
from typing import Dict, Literal, Optional
import torch
import lightning as L

from litgpt import GPT, Config
from litgpt.utils import load_checkpoint
from tqdm import tqdm


class LitGPT(GPT):

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        model_name_or_path = Path(model_name_or_path)
        if not model_name_or_path.is_dir():
            model_name_or_path = Path(os.getenv("LIT_CHECKPOINTS")) / model_name_or_path
            if not model_name_or_path.is_dir():
                raise ValueError(f"Invalid model_name_or_path {model_name_or_path}")
        config = Config.from_checkpoint(model_name_or_path)
        super().__init__(config)

    def init_params(self, fabric: L.Fabric):
        checkpoint_path = Path(self.model_name_or_path) / "lit_model.pth"
        if not checkpoint_path.is_file():
            checkpoint_path = Path(os.getenv("LIT_CHECKPOINTS")) / self.model_name_or_path / "lit_model.pth"
        load_checkpoint(fabric, self, checkpoint_path, strict=False)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        
        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        
        x = self.transformer.wte(idx)  # token embeddings of shape (1, T, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x) # (1, T, n_embd)
        outputs = {
            "logits": self.lm_head(x)
        }

        if output_last_hidden_state:
            outputs["last_hidden_state"] = x
            
        return outputs


class LanguageModelForClassification(L.LightningModule):

    def __init__(
        self, 
        language_model,
        loss_fn: Literal["cross_entropy"] = "cross_entropy",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        global_batch_size: int = 2,
        micro_batch_size: int = 1
    ):
        super().__init__()
        self.lm = language_model
        self.lm.set_kv_cache(batch_size=1)

        # Training arguments
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, output_last_hidden_state: bool = True) -> torch.Tensor:
        return self.lm(idx, input_pos, output_last_hidden_state)
    
    def on_train_start(self) -> None:
        self.gradient_accumulation_steps = (self.global_batch_size // self.fabric.world_size) // self.micro_batch_size
        if self.loss_fn != "cross_entropy":
            raise NotImplementedError(f"Loss function {self.loss_fn} not implemented")
        self.train()
    
    def training_step(
        self,
        batch: Dict[Literal["prompt_ids", "prompt_mask", "answers_ids", "labels"], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        answers_ids = batch["answers_ids"]
        labels = batch["labels"]

        loss = 0
        num_tokens = 0
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
        loss = loss / num_tokens
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.lm.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def on_validation_start(self) -> None:
        self.eval()

    def validation_step(
        self,
        batch: Dict[Literal["prompt_ids", "prompt_mask", "answers_ids", "labels"], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        answers_ids = batch["answers_ids"]
        labels = batch["labels"]

        loss = 0
        num_tokens = 0
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
        return {"cum_loss": loss, "num_tokens": num_tokens}
    

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.prompts_ids = torch.tensor([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]
        ])
        self.prompt_mask = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1]
        ])
        self.answers_ids = [
            torch.tensor([25, 26, 27, 28, 29]),
            torch.tensor([30, 31, 32, 34]),
            torch.tensor([35, 38, 39]),
            torch.tensor([45, 46, 47, 48, 49])
        ]
        self.labels = [0, 3, 2, 1, 1]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "prompt_ids": self.prompts_ids[idx].unsqueeze(0),
            "prompt_mask": self.prompt_mask[idx].unsqueeze(0),
            "answers_ids": self.answers_ids,
            "labels": self.labels[idx]
        }

def collate_fn(batch):
    return {
        "prompt_ids": torch.cat([b["prompt_ids"] for b in batch], dim=0),
        "prompt_mask": torch.cat([b["prompt_mask"] for b in batch], dim=0),
        "answers_ids": [b["answers_ids"] for b in batch],
        "labels": torch.tensor([b["labels"] for b in batch])
    }

def main():

    # General
    seed = 8392

    # Model
    model_name_or_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T-bfloat16"
    model_class = LitGPT

    # Data
    # TODO: Add data

    # Training
    accelerator = "gpu"
    precision = "bf16-true"
    devices = 1
    num_epochs = 10
    batch_size = 2
    micro_batch_size = 1
    loss_fn = "cross_entropy"
    learning_rate = 1e-3
    weight_decay = 0.0
    eval_every_n_steps = 5

    # Initialize the fabric
    fabric = L.Fabric(accelerator=accelerator, precision=precision, devices=devices)
    fabric.seed_everything(seed)
    
    # Initialize the model
    with fabric.init_module():
        language_model = model_class(model_name_or_path)
    language_model.init_params(fabric)
    model = LanguageModelForClassification(language_model, loss_fn, learning_rate, weight_decay, batch_size, micro_batch_size)
    
    # Get the optimizer
    optimizer = model.configure_optimizers()
    optimizer.zero_grad()

    # Get the dataloader
    dataset = DummyDataset()
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Set up objects
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # Train the model
    step_count = 0
    model.on_train_start()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            is_accumulating = (batch_idx + 1) % model.gradient_accumulation_steps != 0
            # TODO: Add gradient accumulation

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                train_loss = model.training_step(batch, batch_idx)
                fabric.backward(train_loss)
            print(f"Epoch {epoch} | Step {step_count} | Train Loss {train_loss.item():.4f}")

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

            if step_count % eval_every_n_steps == 0:
                model.on_validation_start()
                cum_loss = cum_tokens = 0
                progress_bar = tqdm(val_dataloader, leave=False, dynamic_ncols=True, desc="Validation")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(progress_bar):
                        out = model.validation_step(batch, batch_idx)
                        cum_loss += out["cum_loss"].item()
                        cum_tokens += out["num_tokens"]
                    val_loss = cum_loss / cum_tokens
                print(f"Epoch {epoch} | Step {step_count} | Validation Loss {val_loss:.4f}")
                model.on_train_start()

            step_count += 1
            

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    main()
        