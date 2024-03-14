
import os
from lightning.fabric.loggers import TensorBoardLogger
from ...utils import save_yaml, load_yaml


class TBLogger(TensorBoardLogger):

    def __init__(self, root_dir, version):
        super().__init__(
            root_dir=root_dir,
            name="",
            version=version,
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


class ModelCheckpoint:

    def __init__(self, fabric, model_checkpoint_dir, hyperparams):
        self.fabric = fabric
        self.model_checkpoint_dir = model_checkpoint_dir
        self.hyperparams = hyperparams

    def find_last_version(self, state):

        # Try to start from last version
        versions = [int(d.split("version_")[-1]) for d in os.listdir(self.model_checkpoint_dir) if d.startswith("version_")]
        if not versions:
            state["model"].init_params(self.fabric)
            version = "version_0"
        else:
            last_version = max(versions)
            if os.path.exists(os.path.join(self.model_checkpoint_dir, f"version_{last_version}", "training.success")):
                state["model"].init_params(self.fabric)
                version = f"version_{last_version+1}"
                os.makedirs(os.path.join(self.model_checkpoint_dir, version), exist_ok=True)
            else:
                version = f"version_{last_version}"
                hyperparms = load_yaml(os.path.join(self.model_checkpoint_dir, version, "hyperparams.yaml"))
                if hyperparms != self.hyperparams:
                    raise ValueError(f"Hyperparameters mismatch: {hyperparms} != {self.hyperparams}")                
                with open(os.path.join(self.model_checkpoint_dir, version, "training.interrupted"), "r") as f:
                    state["offset_time"] = float(f.read())
                self.fabric.load(os.path.join(self.model_checkpoint_dir, version, "last_model.ckpt"), state)
    
        return state, version