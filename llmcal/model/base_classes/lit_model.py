from typing import Optional
import torch
from litgpt import GPT


class LitGPT(GPT):

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