
from . import prompts

def load_prompt(config: dict):
    prompt_cls_name = config.pop("class_name")
    prompt_cls = getattr(prompts, prompt_cls_name)
    prompt = prompt_cls(**config)
    return prompt