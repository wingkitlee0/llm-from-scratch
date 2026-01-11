import os
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import lightning as L
import ray
import ray.train
import torch
from ray.train import Checkpoint


@dataclass
class IntervalConfig:
    every_n_batches: Optional[int] = None
    every_n_epochs: Optional[int] = 1

    batch_idx: int = field(init=False, default=0)  # non-persistent state
    epoch_idx: int = field(init=False, default=0)  # non-persistent state

    @contextmanager
    def on_train_batch_end(self) -> Iterator[bool]:
        if self.every_n_batches is None:
            should_run = False
        elif self.batch_idx > 0 and self.batch_idx % self.every_n_batches == 0:
            should_run = True
        else:
            should_run = False

        try:
            yield should_run
        finally:
            self.batch_idx += 1

    @contextmanager
    def on_train_epoch_end(self, current_epoch) -> Iterator[bool]:
        if self.every_n_epochs is None:
            should_run = False
        elif current_epoch % self.every_n_epochs == 0:
            should_run = True
        else:
            should_run = False

        try:
            yield should_run
        finally:
            self.epoch_idx += 1
            self.batch_idx = 0


class PrintMixin:
    def print(self, *args, **kwargs):
        if len(args) == 0:
            return

        prefix = f"[{self.__class__.__name__}]"
        print(prefix, *args, **kwargs)


class CustomRayTrainReportCallback(L.Callback, PrintMixin):
    """A callback that reports checkpoints to Ray at custom frequencies.

    Args:
        interval: The interval configuration for saving checkpoints.
    """

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(
        self,
        interval: IntervalConfig,
    ) -> None:
        super().__init__()
        self.interval = interval
        # Track last checkpoint to avoid duplicates
        self._last_checkpointed_global_step = -1
        self._checkpoint_id = 0

        job_id = ray.get_runtime_context().get_job_id()
        experiment_name = ray.train.get_context().get_experiment_name()
        self.local_rank = ray.train.get_context().get_local_rank()

        self.tmpdir_prefix = Path(
            tempfile.gettempdir(),
            f"lightning_checkpoints-job_id={job_id}-name={experiment_name}",
        ).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

    def _save_and_report(
        self,
        trainer: L.Trainer,
        checkpoint_id: str | None = None,
    ) -> None:
        """Save checkpoint and report to Ray Train.

        Args:
            trainer: The Lightning trainer instance.
            checkpoint_id: The ID of the checkpoint. If None, a random UUID will be generated.
            save_checkpoint: If True, save checkpoint to local filesystem.
        """
        if checkpoint_id is None:
            checkpoint_id = f"step_{trainer.global_step}_{str(uuid.uuid4())[:8]}"

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Save checkpoint to local filesystem
        tmpdir = Path(self.tmpdir_prefix, checkpoint_id).as_posix()
        os.makedirs(tmpdir, exist_ok=True)
        ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = Checkpoint.from_directory(tmpdir)

        ray.train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)

        self._checkpoint_id += 1

    def on_train_start(self, trainer, pl_module) -> None:
        if trainer.global_step > 0:
            # Update last checkpointed step to prevent duplicates
            self.print("Resuming from step", trainer.global_step)
            self._last_checkpointed_global_step = trainer.global_step

    def state_dict(self) -> dict:
        """Save callback state to checkpoint."""
        return {
            "_last_checkpointed_global_step": self._last_checkpointed_global_step,
            "every_n_batches": self.interval.every_n_batches,
            "every_n_epochs": self.interval.every_n_epochs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore callback state from checkpoint."""
        self._last_checkpointed_global_step = state_dict.get(
            "_last_checkpointed_global_step", -1
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """
        Mid-epoch checkpointing based on batch count (since start/resume).
        """
        with self.interval.on_train_batch_end() as should_run:
            if not should_run:
                return

            self.print(
                "Saving checkpoint at batch",
                batch_idx,
                "global step",
                trainer.global_step,
            )
            self._save_and_report(
                trainer,
                checkpoint_id=f"batch_{batch_idx}_step_{trainer.global_step}",
            )
            self._last_checkpointed_global_step = trainer.global_step

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        with self.interval.on_train_epoch_end(trainer.current_epoch) as should_run:
            if not should_run:
                return

            if trainer.global_step == self._last_checkpointed_global_step:
                self.print(
                    "Skipping checkpointing at epoch end because we already checkpointed at this training step"
                )
                return

            self._save_and_report(
                trainer,
                checkpoint_id=f"epoch_{trainer.current_epoch}",
            )
            self._last_checkpointed_global_step = trainer.global_step


class TrainLossLoggingCallback(L.Callback, PrintMixin):
    """A callback that logs train loss to MLFlow at custom frequencies.

    Args:
        every_n_steps: Log train loss every N training steps. If None, disabled.
        verbose: If True, print metrics to console when logging. Defaults to False.
    """

    def __init__(self, interval: IntervalConfig, verbose: bool = False) -> None:
        """
        Args:
            interval: The interval configuration for logging train loss.
            verbose: If True, print metrics to console when logging. Defaults to False.
        """
        super().__init__()
        self.interval = interval
        self.verbose = verbose

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Called at the end of each training batch.

        Note: This is called AFTER training_step but AFTER optimizer step only when
        accumulate_grad_batches condition is met.
        """
        with self.interval.on_train_batch_end() as should_run:
            if not should_run:
                return

            # Get train_loss from callback_metrics
            if "train_loss" in trainer.callback_metrics:
                train_loss = trainer.callback_metrics["train_loss"]
                # Convert to float for printing if it's a tensor
                train_loss_value = float(
                    train_loss.item()
                    if hasattr(train_loss, "item")
                    else float(train_loss)
                )

                # Log using pl_module.log() which automatically routes to all loggers (MLFlow, etc.)
                # Use a custom metric name to distinguish from default train_loss logging
                pl_module.log(
                    "train_loss_custom", train_loss, on_step=True, logger=True
                )
                ray.train.report(metrics={"train_loss_custom": train_loss_value})

                # Print metrics if verbose is enabled
                if self.verbose:
                    self.print(
                        f"Batch {batch_idx} global step {trainer.global_step}: train_loss_custom = {train_loss_value:.4f}"
                    )
            else:
                self.print(
                    f"Warning: 'train_loss' not found in callback_metrics at batch {batch_idx} global step {trainer.global_step}"
                )
                self.print(
                    f"  Available metrics: {list(trainer.callback_metrics.keys())}"
                )

    def on_validation_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items() if "val" in k}
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step
        ray.train.report(metrics=metrics)


class TimingDebugCallback(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.validation_idx = 0

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.start_time = time.perf_counter()
        torch.cuda.empty_cache()

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.end_time = time.perf_counter()
        print(
            f"Validation-{self.validation_idx} taken: {self.end_time - self.start_time:.2f} seconds"
        )
        self.validation_idx += 1

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.batch_start_time = time.perf_counter()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.batch_end_time = time.perf_counter()
        print(
            f"Validation-{self.validation_idx} batch-{batch_idx} time: {self.batch_end_time - self.batch_start_time:.2f} seconds"
        )

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.train_batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.train_batch_end_time = time.perf_counter()
        print(
            f"Train-{trainer.global_step} batch-{batch_idx} time: {self.train_batch_end_time - self.train_batch_start_time:.2f} seconds"
        )
