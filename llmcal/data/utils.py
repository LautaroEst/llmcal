import os
from datasets import load_from_disk, Dataset
import numpy as np
from typing import Literal, Union, Any


def init_casting_obj(data: Dataset, cast_config: dict) -> Dataset:
    from .. import casting
    cast_cls_name = cast_config.pop("class_name")
    cast_cls = getattr(casting, cast_cls_name)
    cast = cast_cls(**cast_config)
    cast.fit(data)
    return cast


def load_dataset_and_cast_task(
    dataset: str, 
    split: Literal["train", "validation", "test"],
    n_samples: int = None,
    random_state: int = None,
    cast_obj_or_config: Union[dict,Any] = {},
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
    if isinstance(cast_obj_or_config, dict):
        cast_obj_or_config = init_casting_obj(data, cast_obj_or_config)
    data = cast_obj_or_config.transform(data).select_columns(["idx", "input", "target"])
    return data, cast_obj_or_config