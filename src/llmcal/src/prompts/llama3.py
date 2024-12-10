

class Llama3Prompt:

    def __init__(self, max_characters=400):
        self.max_characters = max_characters
        self.prompt = None

    def apply(self, text):
        filled_prompts = []
        for t in text:
            filled_prompts.append(self.prompt.replace("{inpt}", t[:self.max_characters]))
        return filled_prompts
    
    def fit(self, prompt_template, shots):
        preface = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{prompt_template}<|eot_id|>" # No newline
        )
        output_preface = (
            "<|start_header_id|>user<|end_header_id|>\n\n{inpt}<|eot_id|>" # No newline
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        if len(shots) == 0:
            self.prompt = preface + output_preface
            return self
        
        shot_template = (
            "<|start_header_id|>user<|end_header_id|>\n\n{shot_inpt}<|eot_id|>" # No newline
            "<|start_header_id|>assistant<|end_header_id|>\n\n{shot_label}<|eot_id|>" # No newline
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self