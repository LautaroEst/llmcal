import torch
from torch.utils.data import DataLoader, RandomSampler
from .utils import Template


class ClassificationTemplateCollator:

    def __init__(self, tokenizer, template, labels):
        self.tokenizer = tokenizer
        self.template = Template(prompt=template)
        self.labels = labels
        self.encoded_labels = [self.tokenizer(label_name, return_tensors="pt", padding=True, truncation=False).input_ids for label_name in self.labels]
        self.max_length = self.tokenizer.model_max_length - max([label.shape[1] for label in self.encoded_labels])

    def __call__(self, batch):
        indices, prompts, labels = [], [], []
        features = {feature: [] for feature in self.template.features}
        for sample in batch:
            indices.append(sample["idx"])
            features_dict = {feature: value for feature, value in sample.items() if feature in self.template.features}
            for feature, value in features_dict.items():
                features[feature].append(value)
            prompt = self.template.construct_prompt(**features_dict)
            prompts.append(prompt)
            labels.append(sample["label"])
        encoded_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)

        return {
            "idx": indices,
            "input_ids": encoded_prompts["input_ids"],
            "attention_mask": encoded_prompts["attention_mask"],
            "label": torch.tensor(labels, dtype=torch.long),
            "encoded_labels": [l.clone() for l in self.encoded_labels],
            "features": features,
        }


class LoaderWithTemplate(DataLoader):

    def __init__(self, dataset, template, labels, tokenizer, batch_size=32, shuffle=False, random_state=0, **kwargs):
        self.tokenizer = tokenizer
        self.template = template
        kwargs.pop("collate_fn", None)
        kwargs.pop("sampler", None)
        if shuffle:
            generator = torch.Generator().manual_seed(random_state)
            sampler = RandomSampler(dataset, replacement=False, num_samples=None, generator=generator)
        else:
            sampler = None
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=ClassificationTemplateCollator(
                tokenizer=tokenizer,
                template=template,
                labels=labels,
            ),
            **kwargs
        )