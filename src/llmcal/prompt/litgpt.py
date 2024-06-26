import re

class LitGPTPrompt:

    def __init__(self, preshots_template, shots_template, postshots_template, answers_templates):
        self.preshots_template = preshots_template
        self.shots_template = shots_template
        self.postshots_template = postshots_template
        self.answers_templates = answers_templates

    def fit(self, dataset = []):
        self.preshots_features = re.findall(r'\{(\w+)\}', self.preshots_template)
        self.shots_features = re.findall(r'\{(\w+)\}', self.shots_template)
        self.postshots_features = re.findall(r'\{(\w+)\}', self.postshots_template)
        self.answers_features = [re.findall(r'\{(\w+)\}', answer) for answer in self.answers_templates]

        shot_prompt = ""
        for i in range(len(dataset)):
            shot = dataset[i]
            shot["answer"] = self.answers_templates[shot["label"]].format(**{feature: shot[feature] for feature in self.answers_features[shot["label"]]})
            shots_features = {feature: shot[feature] for feature in self.shots_features}
            filled_shot = self.shots_template.format(**shots_features)
            shot_prompt += filled_shot
        self.shot_prompt = shot_prompt
        return self

    def transform(self, **kwargs):
        preshots_features = {feature: kwargs[feature] for feature in self.preshots_features}
        preshots_prompt = self.preshots_template.format(**preshots_features)
        postshots_features = {feature: kwargs[feature] for feature in self.postshots_features}
        postshots_prompt = self.postshots_template.format(**postshots_features)
        prompt = f"{preshots_prompt}{self.shot_prompt}{postshots_prompt}"
        answers = [answer.format(**{feature: kwargs[feature] for feature in features}) for answer, features in zip(self.answers_templates, self.answers_features)]
        return {
            "prompt": prompt,
            "answers": answers,
        }
    

if __name__ == "__main__":
    dataset = [
        {"animal": "cat", "size": "little", "label": 1},
        {"animal": "dog", "size": "big", "label": 0},
        {"animal": "bird", "size": "small", "label": 1},
        {"animal": "fish", "size": "tiny", "label": 0},
    ]
    prompt = LitGPTPrompt(
        preshots_template="What is the color of the {animal}?\n",
        shots_template="The color of the {animal} is {answer}.",
        postshots_template="\nWhat is the color of the {animal}? ",
        answers_templates=["Black and {size}", "White"]
    )
    prompt.fit(dataset[:3])
    result = prompt.transform(**dataset[3])
    print(result["prompt"])
    print(result["answers"])