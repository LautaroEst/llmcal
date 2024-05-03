from litgpt.lora import LoRALinear

    
def init_lora_linear_modules(module):
    if isinstance(module, LoRALinear):
        module.reset_parameters()
    else:
        for child in module.children():
            init_lora_linear_modules(child)