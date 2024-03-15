
import os
from lightning.fabric.loggers import TensorBoardLogger
from ...utils import save_yaml


class TBLogger(TensorBoardLogger):

    def __init__(self, root_dir):
        _root_dir = "/".join(root_dir.split("/")[:-1])
        _version = root_dir.split("/")[-1]
        super().__init__(
            root_dir=_root_dir,
            name="",
            version=_version,
            default_hp_metric=False
        )

    def log_hyperparams(self, hyperparams, metrics = None):
        super().log_hyperparams(hyperparams, metrics)
        save_yaml(hyperparams, os.path.join(self.log_dir, "hyperparams.yaml"))

    def finalize(self, status, time):
        with open(os.path.join(self.root_dir, self.version, f"training.{status}"), "w") as f:
            f.write(str(time))
        if status == "success":
            if os.path.exists(os.path.join(self.root_dir, self.version, "training.interrupted")):
                os.remove(os.path.join(self.root_dir, self.version, "training.interrupted"))
        super().finalize(status)

