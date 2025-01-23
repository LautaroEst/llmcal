


class GemmaPrompt:

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
            "<start_of_turn>user\n"
            f"{prompt_template}\n\n"
        )
        output_preface = (
            "{inpt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        
        if len(shots) == 0:
            self.prompt = preface + output_preface
            return self
        
        shot_template = (
            "{shot_inpt}<end_of_turn>\n"
            "<start_of_turn>model\n{shot_label}<end_of_turn>\n<start_of_turn>user\n"
        )
        shots_prompt = ""
        for shot in shots:
            shots_prompt += shot_template.format(shot_inpt=shot["text"][:self.max_characters], shot_label=shot["label"])
        self.prompt = preface + shots_prompt + output_preface

        return self