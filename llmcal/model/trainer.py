import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm
from ..utils import save_yaml
from collections import defaultdict


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



class Trainer:

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = 1,
        num_nodes: int = 1,
        precision: Union[str, int] = None,
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        random_state: int = 0,
    ) -> None:
        
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=TBLogger(root_dir=checkpoint_dir),
        )
        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf") or limit_train_batches is None

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf") or limit_val_batches is None

        self.limit_train_batches = limit_train_batches if isinstance(limit_train_batches, int) else float("inf")
        self.limit_val_batches = limit_val_batches if isinstance(limit_val_batches, int) else float("inf")
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}
        self.best_state = {"val_loss": float("inf"), "epoch": 0, "global_step": 0}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        self.random_state = random_state

    def fit(self, model: L.LightningModule):

        # Launch training
        self.fabric.launch()
        self.fabric.seed_everything(self.random_state)

        # Process data and load dataloaders
        with self.fabric.rank_zero_first():
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            model.prepare_data()
        model.setup(stage="fit")
        train_loader = model.train_dataloader()
        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        val_loader = model.val_dataloader()
        val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)

        # Init model's weights
        model.fabric = self.fabric
        model.configure_model()
        model = self.fabric.setup_module(model)

        # setup model and optimizer
        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        if optimizer is not None:
            optimizer = self.fabric.setup_optimizers(optimizer)
        else:
            self.should_stop = False # reset for next fit call
            self.fabric.print("No optimizer provided. Skipping training...")
            return

        # try to load checkpoint
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}
        if os.path.exists(os.path.join(self.checkpoint_dir, "checkpoint.ckpt")):
            self.load(state, "checkpoint.ckpt")
        model, optimizer, scheduler_cfg = state["model"], state["optim"], state["scheduler"]

        # check if we even need to train here
        if (self.max_epochs is not None and self.current_epoch >= self.max_epochs) or (self.max_steps is not None and self.global_step >= self.max_steps):
            self.should_stop = False # reset for next fit call
            self.fabric.print("Found already trained model. Skipping training...")
            return

        while not self.should_stop:

            # Main training loop (1 epoch)
            self.train_loop(
                model, optimizer, 
                train_loader=train_loader, limit_batches=self.limit_train_batches, 
                val_loader=val_loader, limit_val_batches=self.limit_val_batches,
                scheduler_cfg=scheduler_cfg
            )

            # Step scheduler on epoch level
            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

            # increase epoch count
            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        # reset for next fit call
        self.should_stop = False

        # Save when finished
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}
        self.save(state, is_best=self.best_state["val_loss"] > model.avg_val_loss)

        self.fabric.call("on_fit_end")

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        limit_val_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.fabric.call("on_train_epoch_start")
        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # check if optimizer should step in gradient accumulation
            should_optim_step = (batch_idx + 1) % self.grad_accum_steps == 0 or batch_idx == len(train_loader) - 1
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer)

                # optimizer step runs train step internally through closure
                if model.automatic_optimization:
                    optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx))
                    self.fabric.call("on_before_zero_grad", optimizer)
                    optimizer.zero_grad()
                else:
                    outputs = model.training_step(batch, batch_idx=batch_idx)
                    self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return, "train")

            if self.should_validate and should_optim_step:
                self.val_loop(model, val_loader, limit_batches=limit_val_batches)

            if self.should_checkpoint and should_optim_step:
                state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}
                self.save(state, is_best=self.best_state["val_loss"] > model.avg_val_loss)

            # only increase global step if optimizer stepped
            if not model.automatic_optimization:
                self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end")

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        if val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model)):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation", leave=False)

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx, 0)

            out = model.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx, 0)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(self, model: L.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch, batch_idx=batch_idx)

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        return loss

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir
    
    @checkpoint_dir.setter
    def checkpoint_dir(self, value: str):
        self._checkpoint_dir = value
        self.fabric.checkpoint_dir = value
    
    @property
    def global_step(self):
        return self._global_step
    
    @global_step.setter
    def global_step(self, value: int):
        self._global_step = value
        self.fabric.global_step = value

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.global_step % self.validation_frequency == 0
    
    @property
    def should_checkpoint(self) -> bool:
        """Whether to currently run checkpointing."""
        return self.global_step % self.checkpoint_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], ckpt_name: Optional[str] = "checkpoint.ckpt") -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
        """
        if state is None:
            state = {}

        path = os.path.join(self.checkpoint_dir, ckpt_name)
        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")
        self.best_state = remainder.pop("best_state")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping], is_best: Optional[bool] = False) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        if is_best:
            self.best_state = {
                "val_loss": state["model"].avg_val_loss,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            }

        state.update(global_step=self.global_step, current_epoch=self.current_epoch, best_state=self.best_state)
        self.fabric.save(os.path.join(self.checkpoint_dir, f"checkpoint.ckpt"), state)

        if is_best:
            self.fabric.save(os.path.join(self.checkpoint_dir, f"best.ckpt"), state)
            
    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, L.fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    def predict(self, model):

        model = self.fabric.setup_module(model)
        model.setup(stage="predict")

        torch.set_grad_enabled(False)

        self.fabric.call("on_predict_start")

        predict_dataloaders = model.predict_dataloader()
        for dataloader_idx, dataloader in predict_dataloaders.items():
            dataloader = self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=self.use_distributed_sampler)

            self.fabric.call("on_predict_epoch_start")

            outputs = defaultdict(list)
            iterable = self.progbar_wrapper(dataloader, total=len(dataloader), desc=f"Predicting {dataloader_idx}")
            for batch_idx, batch in enumerate(iterable):

                self.fabric.call("on_predict_batch_start", batch, batch_idx, dataloader_idx)
                out = model.predict_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
                self.fabric.call("on_predict_batch_end", out, batch, batch_idx, dataloader_idx)
                for k, v in out.items():
                    outputs[k].append(v.cpu())

            self.fabric.call("on_predict_epoch_end")
            for k, v in outputs.items():
                outputs[k] = torch.cat(v, dim=0)
            torch.save(outputs, os.path.join(self.checkpoint_dir, f"predict_{dataloader_idx}.pt"))

        self.fabric.call("on_predict_end")
        
        torch.set_grad_enabled(True)
        



