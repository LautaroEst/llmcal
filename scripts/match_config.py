
import os
from llmcal.utils import load_yaml, save_yaml
from collections.abc import MutableMapping

def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def load_matching_experiments(**kwargs):
    matching_experiments = []
    for task in os.listdir("experiments"):
        for model in os.listdir(os.path.join("experiments", task)):
            for splits in os.listdir(os.path.join("experiments", task, model)):
                config = load_yaml(os.path.join("experiments", task, model, splits, "config.yaml"))
                flatten_config = flatten(config, separator=".")
                if all([flatten_config[key] == value for key, value in kwargs.items()]):
                    matching_experiments.append((task, model, splits))
    return matching_experiments

    

def main(
    **kwargs
):
    experiments = load_matching_experiments(**kwargs)
    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)