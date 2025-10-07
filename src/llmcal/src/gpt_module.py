from typing import List, Optional, Union
from litgpt.model import GPT as BaseModel, batched_index_select, partial, do_softcapping, nn
import torch

class GPT(BaseModel):
    def forward(
        self,
        idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
        lm_head_chunk_size: int = 0,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        If `input_pos` is provided, the KV cache uses K and V vectors for
        positions smaller than entries in `input_pos`. For efficiency, pass
        `input_pos_maxp1` as `max(input_pos) + 1` if already available from
        your forward algorithm. This slices the KV cache buffers and speeds
        up multi-head attention.

        Without `input_pos_maxp1`, the computation uses the full KV cache
        (`max_seq_length`) with masking applied. Note that inferring
        `input_pos_maxp1` from `input_pos` causes graph breaks and prevents
        compilation.

        Args:
            idx: Token indices of input sequences, shape `(B, T)`, where `B`
                is batch size.
            input_pos: Optional. Positions of input tokens. The default is
                `arange(T)`. Can have shape `(T,)` or `(B, T)` (batched index).
            input_pos_maxp1: Optional. See above.
            lm_head_chunk_size: Optional. If `lm_head_chunk_size > 0`, the final
                `lm_head` computation is done in chunks of this size.

        Returns:
            Logit outputs, shape `(B, T, config.padded_vocab_size)`. If
            `lm_head_chunk_size > 0`, this is a list of chunks of shape
            `(B, lm_head_chunk_size, config.padded_vocab_size)`, the final
            entry can be shorter.

        """
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            if input_pos.dim() > 2:
                # otherwise, things go wrong in `apply_rope`
                raise ValueError(f"input_pos must have 1 or 2 dimensions, input_pos.shape = {input_pos.shape}")
            if input_pos.shape[-1] != T:
                raise ValueError(f"input_pos.shape[-1] = {input_pos.shape[-1]} != {T} = idx.shape[1], must be the same")
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if input_pos.dim() == 1:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.view(*(mask.shape[0:1] + mask.shape[2:]))
            if input_pos_maxp1 is not None:
                # Shorten final dimension so it just covers all `input_pos` entries
                if input_pos_maxp1 > self.max_seq_length:
                    raise ValueError(f"Positions in 'input_pos' must be in [0,{self.max_seq_length})")
                mask = mask[..., :input_pos_maxp1]
        else:
            # unsqueeze to have a batch dimension
            cos = self.cos[:T].unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0)
            # `cos`, `sin` have shape (1, T, config.rope_n_elem)
            mask = None  # defaults to causal mask
            input_pos_maxp1 = None

        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block_idx, block in enumerate(self.transformer.h):
            if self.config.rope_indices is not None:
                x = block(
                    x,
                    cos[..., self.config.rope_indices[block_idx]],
                    sin[..., self.config.rope_indices[block_idx]],
                    mask,
                    input_pos,
                    input_pos_maxp1,
                )
            else:
                x = block(x, cos, sin, mask, input_pos, input_pos_maxp1)
        x = self.transformer.ln_f(x)
        clamp_head = (
            partial(do_softcapping, thresh=self.config.final_logit_softcapping)
            if self.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return x[:,-1,:], [clamp_head(self.lm_head(x_i)) for x_i in x.split(lm_head_chunk_size, dim=1)]
        else:
            return x[:,-1,:], clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)


