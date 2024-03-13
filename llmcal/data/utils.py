import os
from datasets import load_from_disk, Dataset
import numpy as np
from typing import Literal, Union, Any


def init_prompt(data: Dataset, prompt_config: dict) -> Dataset:
    from ..prompt import prompts
    prompt_cls_name = prompt_config.pop("class_name")
    prompt_cls = getattr(prompts, prompt_cls_name)
    prompt = prompt_cls(**prompt_config)
    prompt.fit(data)
    return prompt


def load_dataset_and_cast_task(
    dataset: str, 
    split: Literal["train", "validation", "test"],
    n_samples: int = None,
    random_state: int = None,
    prompt_obj_or_config: Union[dict,Any] = {},
) -> Dataset:
    
    # Load dataset and sample
    data = load_from_disk(os.path.join(dataset, split))
    if n_samples is None:
        n_samples = len(data)
    if random_state is not None:
        rs = np.random.RandomState(random_state)
        idx = rs.choice(len(data), n_samples, replace=False).tolist()
        data = data.select(idx)

    # Cast task
    if isinstance(prompt_obj_or_config, dict):
        if len(prompt_obj_or_config) == 0:
            data = data.rename_column("input", "old_input")
            data = data.rename_column("output", "input")
            data = data.remove_columns("old_input")
            return data, prompt_obj_or_config
        prompt_obj_or_config = init_prompt(data, prompt_obj_or_config)
    data = prompt_obj_or_config.transform(data)
    data = data.select_columns(["idx", "input", "label"])
    return data, prompt_obj_or_config