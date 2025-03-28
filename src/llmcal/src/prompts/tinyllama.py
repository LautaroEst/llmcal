


class TinyLlamaPrompt:

    def __init__(self, max_characters=400):
        self.max_characters = max_characters
        self.prompt = None

    def apply(self, text):
        filled_prompts = []
        for t in text:
            filled_prompts.append(self.prompt.format(inpt=t[:self.max_characters]))
        return filled_prompts
    
    def fit(self, prompt_template, shots):
        preface = (
            "<|system|>\n"
            f"{prompt_template}</s>\n"
        )
        output_preface = (
            "<|user|>\n{inpt}</s>\n"
            "<|assistant|>\n"
        )
        
        if len(shots) == 0:
            self.prompt = preface + output_preface
            return self
        
        shot_template = (
           "<|user|>\n{shot_inpt}</s>\n"
            "<|assistant|>\n{shot_label}</s>\n"
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self
    