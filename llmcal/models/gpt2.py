
from typing import List, Optional, Tuple
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer as _GPT2Tokenizer, 
    DynamicCache
)

class GPT2LMClassifier(GPT2LMHeadModel):

    supported_models = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "distilgpt2"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.pad_token_id = self.config.eos_token_id
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_embeddings: Optional[bool] = None,
        encoded_labels: Optional[List[torch.LongTensor]] = None
    ) -> torch.FloatTensor:
        
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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
                    attention_mask=torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (batch_size, label_len),
                                dtype=torch.long,
                                device=attention_mask.device
                            )
                        ),
                        dim=1
                    ),
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
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
        

class GPT2Tokenizer(_GPT2Tokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_side = "left"
        self.pad_token = self.eos_token

    