
from typing import List, Optional, Tuple
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer as _LlamaTokenizer,
    DynamicCache
)


class LlamaLMClassifier(LlamaForCausalLM):

    supported_models = [
        "meta-llama/Llama-2-7b-hf"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.pad_token_id = self.config.eos_token_id
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_embeddings: Optional[bool] = None,
        encoded_labels: Optional[List[torch.LongTensor]] = None
    ) -> torch.FloatTensor:
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=encoded_labels is not None,
            output_attentions=output_attentions,
            output_hidden_states=output_embeddings,
            return_dict=True
        )

        if encoded_labels is not None:
            batch_size = input_ids.shape[0]
            sequence_lens = attention_mask.sum(dim=-1,keepdim=True)
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
            labels_logits = []
            for encoded_label in encoded_labels:
                label_len = encoded_label.shape[1]
                encoded_label_batch = encoded_label.repeat(batch_size,1)
                position_ids = torch.arange(
                    label_len, 
                    device=sequence_lens.device
                ).repeat(batch_size,1) + sequence_lens
                
                logits = super().forward(
                    input_ids=encoded_label_batch,
                    past_key_values=past_key_values,
                    attention_mask=torch.ones((batch_size, label_len), dtype=torch.long, device=attention_mask.device),
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    output_embeddingshidden_states=False,
                    return_dict=True
                ).logits
                
                logits = torch.cat((outputs.logits[:,-1,:].unsqueeze(1),logits[:,:-1,:]),dim=1)
                gathered_logprobs = torch.gather(
                    torch.log_softmax(logits, dim=-1),
                    dim=-1,
                    index=encoded_label_batch.unsqueeze(-1)
                ).squeeze(-1).sum(dim=1)
                labels_logits.append(gathered_logprobs)
            labels_logits = torch.stack(labels_logits,dim=1)
        
        else:
            labels_logits = outputs.logits[:,-1,:]
        
        output = {"logits": labels_logits}
        if output_attentions:
            output["attentions"] = outputs.attentions
        if output_embeddings:
            output["embeddings"] = outputs.hidden_states[-1][:,-1,:]
        return output
    

class LlamaTokenizer(_LlamaTokenizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_side = "left"
        self.pad_token = self.eos_token