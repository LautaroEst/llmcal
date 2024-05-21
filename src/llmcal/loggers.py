import os
import logging
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger as _CSVLogger
import pandas as pd
from .utils import save_yaml


class TBLogger(TensorBoardLogger):

    def __init__(self, save_dir):
        _save_dir = "/".join(save_dir.split("/")[:-1])
        _version = save_dir.split("/")[-1]
        super().__init__(
            save_dir=_save_dir,
            name="",
            version=_version,
            log_graph=False,
            default_hp_metric=False,
            prefix="",
            sub_dir=None,
        )

    def log_hyperparams(self, hyperparams, metrics = None):
        super().log_hyperparams(hyperparams, metrics)
        save_yaml(hyperparams, os.path.join(self.log_dir, "hyperparams.yaml"))


class CSVLogger(_CSVLogger):

    def __init__(self, save_dir):
        _save_dir = "/".join(save_dir.split("/")[:-1])
        _version = save_dir.split("/")[-1]
        super().__init__(
            save_dir=_save_dir,
            name="",
            version=_version,
            prefix="",
            flush_logs_every_n_steps=1,
        )
        if os.path.exists(os.path.join(self.log_dir, "metrics.csv")):
            self.experiment.metrics = pd.read_csv(os.path.join(self.log_dir, "metrics.csv")).to_dict(orient="records")
        


def setup_loggers(output_dir):
    if not os.path.exists(os.path.join(output_dir, "logs")):
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    for level_name, level in [("info", logging.INFO), ("error", logging.ERROR), ("warning", logging.WARNING)]:
        file_handler = logging.FileHandler(os.path.join(output_dir, f'logs/{level_name}.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            if isinstance(logger, logging.Logger) and logger.getEffectiveLevel() == level:
                logger.addHandler(file_handler)