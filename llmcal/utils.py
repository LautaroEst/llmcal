
from pathlib import Path
from typing import Union
import yaml

def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save_yaml(data: dict, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)