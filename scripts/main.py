
import os
import logging
import lightning as L
from llmcal.utils import load_yaml, setup_logger
from llmcal.model.utils import check_model_type
from llmcal.model.pl_modules import *
from llmcal.data.data_modules import *
from llmcal.data.datasets.utils import SUPPORTED_DATASETS
from llmcal.model.loggers import TBLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from lightning.pytorch.trainer.states import TrainerStatus

def main(
    dataset: SUPPORTED_DATASETS,
    prompt: str,
    data_fold: str,
    model: str,
    method: str,
):
    
    model_config = load_yaml(f"configs/model/{model}.yaml")
    prompt_config = load_yaml(f"configs/prompt/{prompt}.yaml")
    data_fold_config = load_yaml(f"configs/fold/{data_fold}.yaml")
    method_config = load_yaml(f"configs/method/{method}.yaml")
    results_dir = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/{method}"

    if os.path.exists(os.path.join(results_dir, "done.txt")):
        print(f"Experiment {results_dir} already done. Skipping.")
        return
    
    logger = logging.getLogger(__name__)
    logger.info(
        "Running experiment with the following configuration:"
        f"Dataset: {dataset}"
        f"prompt: {prompt}"
        f"data_fold: {data_fold},"
        f"model: {model}"
        f"method: {method}"
    )

    os.makedirs(results_dir, exist_ok=True)
    setup_logger(results_dir)
    seed_everything(method_config.get("random_state", 0))

    checkpoint_freq = method_config.get("checkpoint_frequency", 1)
    if checkpoint_freq is not None:
        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=results_dir,
                filename="best",
                monitor="val_loss",
                save_last=False,
                save_top_k=1,
                save_weights_only=False,
                mode="min",
                every_n_train_steps=checkpoint_freq,
                enable_version_counter=False,
                save_on_train_epoch_end=False,
            )
        ]
    else:
        checkpoint_callbacks = []

    checkpoint_callbacks.append(
        ModelCheckpoint(
            dirpath=results_dir,
            filename="last",
            monitor="epoch",
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="max",
            every_n_epochs=1,
            enable_version_counter=False,
            save_on_train_epoch_end=True,
        )
    )

    # Init trainer
    trainer = L.Trainer(
        accelerator = model_config.get("accelerator", "auto"),
        strategy = model_config.get("strategy", "auto"),
        devices = model_config.get("devices", "auto"),
        num_nodes = model_config.get("num_nodes", 1),
        precision = model_config.get("precision", "32-true"),
        logger = TBLogger(save_dir=results_dir),
        callbacks = checkpoint_callbacks,
        fast_dev_run = method_config.get("fast_dev_run", False),
        max_epochs = method_config.get("max_epochs", 1000),
        min_epochs = method_config.get("min_epochs", 1),
        max_steps = method_config.get("max_steps", -1),
        min_steps = method_config.get("min_steps", 1),
        max_time = method_config.get("max_time", None),
        limit_train_batches = method_config.get("limit_train_batches", None),
        limit_val_batches = method_config.get("limit_val_batches", None),
        limit_test_batches = method_config.get("limit_test_batches", None),
        limit_predict_batches = method_config.get("limit_predict_batches", None),
        overfit_batches = method_config.get("overfit_batches", 0),
        val_check_interval = method_config.get("val_check_interval", 1),
        check_val_every_n_epoch = method_config.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps = method_config.get("num_sanity_val_steps", None),
        log_every_n_steps = None,
        enable_checkpointing = True,
        enable_progress_bar = True,
        enable_model_summary = True,
        accumulate_grad_batches = method_config.get("accumulate_grad_batches", 1),
        gradient_clip_val = None,
        gradient_clip_algorithm = None,
        deterministic = True,
        benchmark = None,
        use_distributed_sampler = True,
        profiler = "simple",
        detect_anomaly = False,
        barebones = False,
        plugins = None,
        sync_batchnorm = False,
        reload_dataloaders_every_n_epochs = 0,
        default_root_dir = results_dir,
    )

    # Init datamodule and model
    task = model_config.pop("task", None)
    method = method_config.pop("method", None)
    model_type = check_model_type(model_config["checkpoint_dir"], method)
    if task == "language_model" and model_type == "litgpt" and method == "full_ft":
        data_dir = f"experiments/{dataset}/.cache"
        data_cache_dir = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache"
        datamodule = LanguageModelLitGPTFineTuningDataModule(
            dataset = dataset,
            data_dir = data_dir,
            data_cache_dir = data_cache_dir,
            tokenizer_dir = model_config["checkpoint_dir"],
            num_train_samples = data_fold_config["train_samples"],
            num_val_samples = data_fold_config["val_samples"], 
            num_shots = data_fold_config["shots_samples"], 
            preshots_template = prompt_config["preshots_template"],
            shots_template = prompt_config["shots_template"],
            postshots_template = prompt_config["postshots_template"],
            shots_separator = prompt_config["shots_separator"],
            answers_templates = prompt_config["answers_templates"],
            batch_size = method_config["batch_size"],
            random_state = data_fold_config["random_state"],
        )
        model_cls = LanguageModelLitGPTFullFT
        model_init_args = dict(
            checkpoint_dir = model_config["checkpoint_dir"],
            embedding_pooling = model_config["embedding_pooling"],
            loss_fn = method_config["loss_fn"],
            optimizer = method_config["optimizer"],
            learning_rate = method_config["learning_rate"],
            weight_decay = method_config["weight_decay"],
        )
    elif task == "language_model" and model_type == "litgpt" and method == "lora":
        data_dir = f"experiments/{dataset}/.cache"
        data_cache_dir = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache"
        datamodule = LanguageModelLitGPTFineTuningDataModule(
            dataset = dataset,
            data_dir = data_dir,
            data_cache_dir = data_cache_dir,
            tokenizer_dir = model_config["checkpoint_dir"],
            num_train_samples = data_fold_config["train_samples"],
            num_val_samples = data_fold_config["val_samples"], 
            num_shots = data_fold_config["shots_samples"], 
            preshots_template = prompt_config["preshots_template"],
            shots_template = prompt_config["shots_template"],
            postshots_template = prompt_config["postshots_template"],
            shots_separator = prompt_config["shots_separator"],
            answers_templates = prompt_config["answers_templates"],
            batch_size = method_config["batch_size"],
            random_state = data_fold_config["random_state"],
        )
        model_cls = LanguageModelLitGPTLoRA
        model_init_args = dict(
            checkpoint_dir = model_config["checkpoint_dir"],
            embedding_pooling = model_config["embedding_pooling"],
            loss_fn = method_config["loss_fn"],
            optimizer = method_config["optimizer"],
            learning_rate = method_config["learning_rate"],
            weight_decay = method_config["weight_decay"],
        )
    elif task == "language_model" and model_type == "litgpt" and method == "no_adaptation":
        data_dir = f"experiments/{dataset}/.cache"
        data_cache_dir = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache"
        datamodule = LanguageModelLitGPTFineTuningDataModule(
            dataset = dataset,
            data_dir = data_dir,
            data_cache_dir = data_cache_dir,
            tokenizer_dir = model_config["checkpoint_dir"],
            num_train_samples = data_fold_config["train_samples"],
            num_val_samples = data_fold_config["val_samples"], 
            num_shots = data_fold_config["shots_samples"], 
            preshots_template = prompt_config["preshots_template"],
            shots_template = prompt_config["shots_template"],
            postshots_template = prompt_config["postshots_template"],
            shots_separator = prompt_config["shots_separator"],
            answers_templates = prompt_config["answers_templates"],
            batch_size = method_config["batch_size"],
            random_state = data_fold_config["random_state"],
        )
        model_cls = LanguageModelLitGPTNoAdaptation
        model_init_args = dict(
            checkpoint_dir = model_config["checkpoint_dir"],
            embedding_pooling = model_config["embedding_pooling"],
        )
    elif task == "language_model" and model_type == "litgpt" and method == "affine_calibration":
        data_dir = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/no_adaptation/"
        data_cache_dir = f"experiments/{dataset}/{data_fold}/{prompt}/{model}/.data_cache"
        if os.path.exists(data_dir):
            data = torch.load(os.path.join(data_dir, "train--logits--predict.pt"))
            num_classes = data.shape[1]
        else:
            raise ValueError(f"Model with no adaptation needs to be run first.")
        datamodule = TensorDataModule(
            data_dir = data_dir,
            data_cache_dir = data_cache_dir,
            num_train_samples = data_fold_config["train_samples"],
            num_val_samples = data_fold_config["val_samples"], 
            random_state = data_fold_config["random_state"],
        )
        model_cls = AffineCalibration
        model_init_args = dict(
            num_classes = num_classes,
            alpha = method_config["alpha"],
            beta = method_config["beta"],
            max_ls = method_config["max_ls"],
        )
    else:
        raise ValueError(f"Invalid combination of task ({task}), checkpoint_dir ({model_config['checkpoint_dir']}) and method ({method})")
    with trainer.init_module():
        model = model_cls(**model_init_args)

    # Fit the model
    last_checkpoint_path = os.path.join(results_dir, "last.ckpt") if os.path.exists(os.path.join(results_dir, "last.ckpt")) else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=last_checkpoint_path)

    if trainer.state.status == TrainerStatus.INTERRUPTED:
        return

    # Predict with the model
    datamodule.setup("predict")
    predict_dataloaders = datamodule.predict_dataloader()
    for i, dataloader in enumerate(predict_dataloaders):
        trainer.predict(model, dataloaders=dataloader)
        if trainer.state.status == TrainerStatus.INTERRUPTED:
            return
        for k, result in model.predict_outputs.items():
            torch.save(result, os.path.join(results_dir, f"{datamodule.idx2split[i]}--{k}--predict.pt"))
    
    with open(os.path.join(results_dir, "done.txt"), "w") as f:
        f.write(results_dir)

if __name__ == "__main__":
    from fire import Fire
    import torch
    torch.set_float32_matmul_precision("high")
    Fire(main)