

class Prompt:

    def __init__(self, prompt_template, answers_templates, tokenizer_dir, max_seq_len):
        self.prompt_template = prompt_template
        self.answers_templates = answers_templates
        self.tokenizer_dir = tokenizer_dir
        self.max_seq_len = max_seq_len