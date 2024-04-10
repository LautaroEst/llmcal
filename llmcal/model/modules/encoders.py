
import torch
from transformers import AutoModelForMaskedLM


class EncoderModel(nn.Module):

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self._model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

    def init_params(self, fabric):
        pass

    def forward(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor):
        output = self._model(
            input_ids=prompt_ids, 
            attention_mask=prompt_mask,
            return_dict=True,
            output_hidden_states=False,
            
        )

