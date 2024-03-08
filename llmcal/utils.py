
from pathlib import Path
from typing import Union
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