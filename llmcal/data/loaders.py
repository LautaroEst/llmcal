import torch
from torch.utils.data import DataLoader, RandomSampler

class ClassificationTemplateCollator:

    MAX_QUERY_TOKENS = 50

    def __init__(self, tokenizer, template):
        self.tokenizer = tokenizer
        self.template = template
        self.labels = template.labels
        num_of_prompt_tokens_without_query = len(
            tokenizer.tokenize(
                template.construct_prompt(**{
                    feature: " " for feature in template.features
                })
            )
        )
        if num_of_prompt_tokens_without_query + self.MAX_QUERY_TOKENS > tokenizer.max_len_single_sentence:
            raise ValueError("The template is too long for the tokenizer.")
        
        max_len_label = max([self.tokenizer.tokenize(l) for l in template.labels], key=len)
        self.max_tokens_per_feature = (
            tokenizer.max_len_single_sentence - num_of_prompt_tokens_without_query - max_len_label - 5
        ) // len(template.features)

    def __call__(self, batch):

        ids, prompts, labels = [], [], []
        for sample in batch:
            ids.append(sample["idx"])
            prompts.append(
                self.template.construct_prompt(**{
                    feature: self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.tokenize(sample[feature])[:self.max_tokens_per_feature]
                    ) for feature in self.template.features
                })
            )
            labels.append(sample["label"])
        encoded_labels = {idx: {k: v.repeat(len(prompts),1) for k, v in self.tokenizer([
            self.template.construct_label(label)
        ], return_tensors="pt", padding=True).items()} for idx, label in enumerate(self.labels)}

        return {
            "idx": ids,
            "prompt": prompts,
            "encoded_prompt": self.tokenizer(prompts, return_tensors="pt", padding=True),
            "label": torch.tensor(labels),
            "encoded_labels": encoded_labels
        }

class LoaderWithTemplateCollator(DataLoader):

    def __init__(self, dataset, template, tokenizer, batch_size=32, shuffle=False, random_state=0, **kwargs):
        self.tokenizer = tokenizer
        self.template = template
        kwargs.pop("collate_fn", None)
        if shuffle:
            generator = torch.Generator().manual_seed(random_state)
            sampler = RandomSampler(dataset, replacement=False, num_samples=None, generator=generator)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=ClassificationTemplateCollator(
                tokenizer=tokenizer,
                template=template
            ),
            **kwargs
        )