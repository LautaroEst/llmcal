

class QwenPrompt:

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
            f"<|im_start|>system\n{prompt_template}<|im_end|>\n"
        )
        output_preface = (
            "<|im_start|>user\n{inpt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        if len(shots) == 0:
            self.prompt = preface + output_preface
            return self
        
        shot_template = (
            "<|im_start|>user\n{shot_inpt}<|im_end|>\n"
            "<|im_start|>assistant\n{shot_label}<|im_end|>\n"
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self