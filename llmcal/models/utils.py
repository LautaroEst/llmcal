
from .gpt2 import GPT2LMClassifier, GPT2Tokenizer
from .llama import LlamaLMClassifier, LlamaTokenizer


SUPPORTED_CLASSES = [
    (GPT2LMClassifier, GPT2Tokenizer),
    (LlamaLMClassifier, LlamaTokenizer)
]


def load_model_and_tokenizer(model_name):

    model = None
    for model_class, tokenizer_class in SUPPORTED_CLASSES:
        if model_name in model_class.supported_models:
            model = model_class.from_pretrained(model_name)
            tokenizer = tokenizer_class.from_pretrained(model_name)
            break
    
    if model is None:
        raise ValueError(f"Model {model_name} not supported.")
    
    return model, tokenizer
