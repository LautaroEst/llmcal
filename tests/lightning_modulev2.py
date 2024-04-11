
import os
from pathlib import Path
from typing import Dict, Literal, Optional
import torch
import lightning as L

from litgpt import GPT, Config
from litgpt.lora import GPT as LoraGPT, Config as LoraConfig, mark_only_lora_as_trainable, LoRALinear
from litgpt.utils import load_checkpoint
from tqdm import tqdm


def init_lora_linear_modules(module):
    if isinstance(module, LoRALinear):
        module.reset_parameters()
    else:
        for child in module.children():
            init_lora_linear_modules(child)

class LoRAGPT(LoraGPT):

    @classmethod
    def from_pretrained(cls, fabric, checkpoint_dir, **kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
            if not checkpoint_dir.is_dir():
                raise ValueError(f"Invalid model_name_or_path {checkpoint_dir}")
        config = LoraConfig.from_checkpoint(checkpoint_dir, **kwargs)
        with fabric.init_module():
            model = cls(config)
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        load_checkpoint(fabric, model, checkpoint_path, strict=False)
        mark_only_lora_as_trainable(model)
        init_lora_linear_modules(model)
        return model

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

class LitGPT(GPT):

    @classmethod
    def from_pretrained(cls, fabric, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = Path(os.getenv("LIT_CHECKPOINTS")) / checkpoint_dir
            if not checkpoint_dir.is_dir():
                raise ValueError(f"Invalid model_name_or_path {checkpoint_dir}")
        config = Config.from_checkpoint(checkpoint_dir)
        with fabric.init_module():
            model = cls(config)
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file {checkpoint_path} not found")
        load_checkpoint(fabric, model, checkpoint_path, strict=False)
        for param in model.parameters():
            param.requires_grad = True
        return model

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
        batch_size: int = 2,
        micro_batch_size: int = 1
    ):
        super().__init__()
        self.lm = language_model
        self.lm.set_kv_cache(batch_size=1)

        # Training arguments
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.global_batch_size = batch_size
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
            [20, 21, 22, 23, 24],
            [1, 1, 2, 3, 4],
            [6, 6, 7, 8, 9],
            [11, 11, 12, 13, 14],
            [16, 16, 17, 18, 19],
            [21, 21, 22, 23, 24],
            [2, 1, 2, 3, 4],
            [7, 6, 7, 8, 9],
            [12, 11, 12, 13, 14],
            [17, 16, 17, 18, 19],
            [22, 21, 22, 23, 24]
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
        self.labels = [0, 3, 2, 1, 1, 4, 4, 2, 1, 1, 2, 1, 2, 1, 1]

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


def init_model(fabric, model_class, checkpoint_dir, **kwargs):
    if model_class in [LitGPT, LoRAGPT]:
        lm = model_class.from_pretrained(fabric, checkpoint_dir)
        model = LanguageModelForClassification(lm, **kwargs)
    else:
        raise ValueError(f"Model class {model_class} not recognized")
            
    return model



def setup(
    accelerator="cpu",
    precision="bf16-true",
    devices=2,
    **kwargs
):
    # Initialize the fabric
    fabric = L.Fabric(accelerator=accelerator, precision=precision, devices=devices)
    fabric.launch(main, **kwargs)


def main(
    fabric,
    seed = 8392,
    model_class = LitGPT,
    checkpoint_dir = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T-bfloat16",
    num_epochs = 10,
    batch_size = 2,
    micro_batch_size = 1,
    loss_fn = "cross_entropy",
    learning_rate = 1e-3,
    weight_decay = 0.0,
    eval_every_n_steps = 5,
):
    
    # Set the seed
    fabric.seed_everything(seed)
    
    # Initialize the model
    model = init_model(
        fabric, 
        model_class, 
        checkpoint_dir,
        loss_fn = loss_fn,
        learning_rate = learning_rate, 
        weight_decay = weight_decay, 
        batch_size = batch_size, 
        micro_batch_size = micro_batch_size
    )
    
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

    iter_train_dataloader = iter(train_dataloader)
    # print(next(iter_train_dataloader))
    # print("finished OK")
    fabric.print("Finished OK")


if __name__ == "__main__":
    from fire import Fire
    torch.set_float32_matmul_precision("high")
    Fire(setup)
    

# TODO: add gradient accumulation
# TODO: data module
# TODO: multiple loggings
# TODO: gin-like configuration
# TODO: extend for Bert Classification

    