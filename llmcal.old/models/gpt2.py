
import torch
from torch import nn
from .base import BaseLMClassifier

from transformers import AutoModelForCausalLM, AutoTokenizer


class GPT2Classifier(BaseLMClassifier):

    SUPPORTED_VERSIONS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    def __init__(self, model_name):

        # Load pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        # Load pretrained model
        if model_name not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Model {model_name} not supported.")
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        base_model.config.pad_token_id = base_model.config.eos_token_id
        
        super().__init__(base_model, tokenizer)

    def forward(self, batch_encoded_prompts, encoded_labels, output_embeddings=False):
        encoder_output = self.prompt_encoder(batch_encoded_prompts, output_embeddings=output_embeddings)
        labels_logits = self.labels_decoder(encoder_output, encoded_labels)
        return encoder_output, labels_logits