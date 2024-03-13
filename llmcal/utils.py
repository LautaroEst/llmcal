
from collections import abc
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml
import time

def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save_yaml(data: dict, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return result
    return wrapper


def update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def perform_modifications(configs: Dict[str,Any], mods: List[str]):
    if len(mods) == 0:
        return configs
    
    for i in range(0,len(mods),2):
        mod = mods[i]
        mod_dict = mods[i+1]
        for m in mod.split(".")[::-1]:
            mod_dict = {m: mod_dict}
    configs = update(configs, mod_dict)
    return configs