
from . import modules, trainers

import torch
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from litgpt.utils import get_default_supported_precision


MODULES_WITH_FABRIC_INIT = [
    "LitGPTLanguageModel",
    "LitGPTPromptClassification",
    "LitGPTSequenceClassification",
    "LoRALitGPTLanguageModel",
    "LoRALitGPTPromptClassification",
    "LoRALitGPTSequenceClassification",
    "AffineCalibrator",
]


def init_fabric(model_args):

    # Configure precision and quantization
    precision = model_args.get("precision", get_default_supported_precision(training=False))
    quantize = model_args.get("quantize", None)
    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    # Configure Callbacks
    callbacks = None

    # Configure Loggers
    loggers = None

    # Init fabric
    fabric = L.Fabric(
        accelerator=model_args.get("accelerator", "auto"),
        strategy=model_args.get("strategy", "auto"),
        devices=model_args.get("devices", "auto"),
        num_nodes=model_args.get("num_nodes", 1),
        precision=precision,
        plugins=plugins,
        callbacks=callbacks,
        loggers=loggers,
    )

    fabric.seed_everything(model_args["train"]["random_state"])

    return fabric


def check_if_trainer_compatile_with_model(trainer_cls_name: str, model_cls_name: str) -> bool:
    if trainer_cls_name == "MiniBatchGDTrainer":
        if model_cls_name in [
            "LitGPTPromptClassification", 
            "LitGPTSequenceClassification", 
            "LitGPTLanguageModel",
            "LoRALitGPTPromptClassification", 
            "LoRALitGPTSequenceClassification", 
            "LoRALitGPTLanguageModel",
            "AffineCalibrator",
            "Linear"
        ]:
            return True
    if trainer_cls_name == "GradientDescentTrainer":
        if model_cls_name in [
            "AffineCalibrator",
            "Linear"
        ]:
            return True
    return False


def load_model(config: dict, model_checkpoint_dir: str = None):
    model_args = config["model"]
    model_cls_name = model_args.pop("class_name")
    model_cls = getattr(modules, model_cls_name)

    if model_cls_name in MODULES_WITH_FABRIC_INIT:
        fabric = init_fabric(config)
        with fabric.init_module(empty_init=True):
            model = model_cls(**model_args)
        model.init_params(fabric)
    else:
        fabric = None
        model = model_cls(**model_args)

    trainer_args = config["train"]
    trainer_args["model_checkpoint_dir"] = model_checkpoint_dir
    trainer_cls_name = trainer_args.pop("class_name")
    if not check_if_trainer_compatile_with_model(trainer_cls_name, model_cls_name):
        raise ValueError(f"Model {model_cls_name} is not compatible with trainer {trainer_cls_name}")
    trainer_cls = getattr(trainers, trainer_cls_name)
    if fabric is None:
        trainer = trainer_cls(**trainer_args)
    else:
        trainer = trainer_cls(fabric, **trainer_args)

    return model, trainer
