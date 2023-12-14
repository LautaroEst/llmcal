

class SubstitutionTemplate:

    def __init__(self, prompt: str, prefix_sample_separator=" "):
        self.prompt = prompt
        self.prefix_sample_separator = prefix_sample_separator
    
    def construct_prompt(self, **kwargs):
        return self.prompt.format(**kwargs)

    def construct_label(self, label: str):
        return f"{self.prefix_sample_separator}{label}"