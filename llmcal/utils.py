import logging
import os
from pathlib import Path
from typing import Union
import yaml

def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save_yaml(data: dict, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


def setup_logger(results_dir):

    loggers = logging.Logger.manager.loggerDict
    for level_name, level in [("info", logging.INFO), ("debug", logging.DEBUG)]:
        file_handler = logging.FileHandler(os.path.join(results_dir, f'{level_name}.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        for logger in loggers:
            if isinstance(loggers[logger], logging.Logger) and loggers[logger].getEffectiveLevel() == level:
                loggers[logger].addHandler(file_handler)
