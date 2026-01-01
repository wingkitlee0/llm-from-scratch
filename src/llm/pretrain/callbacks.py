import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import uuid
import warnings

import lightning as L
import ray
import ray.train
from ray.train import Checkpoint
import time

import torch


class CustomRayTrainReportCallback(L.Callback):
    """A callback that reports checkpoints to Ray at custom frequencies.

    Args:
        every_n_batches: Save checkpoint every N training steps. If None, disabled.
        every_n_epochs: Save checkpoint every N epochs. If None, disabled.
            Defaults to 1 for backward compatibility.
        save_on_train_epoch_end: Whether to save at the end of each epoch.
            Deprecated in favor of every_n_epochs.
    """

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(
        self,
        every_n_batches: Optional[int] = None,
        every_n_epochs: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.every_n_batches = every_n_batches
        self.every_n_epochs = every_n_epochs
        self._last_checkpointed_step = -1  # Track last checkpoint to avoid duplicates
        # Track batches since start/resume (resets on resume, does NOT persist)
        self._session_batch_idx = 0
        # Whether the callback is resumed from a checkpoint
        self.is_first_epoch_since_resumed = False
        # Track the number of checkpoints created in the current callback instance (reset on resume)
        self._current_checkpoint_id = 0

        job_id = ray.get_runtime_context().get_job_id()
        experiment_name = ray.train.get_context().get_experiment_name()
        self.local_rank = ray.train.get_context().get_local_rank()

        self.tmpdir_prefix = Path(
            tempfile.gettempdir(),
            f"lightning_checkpoints-job_id={job_id}-name={experiment_name}",
        ).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if stage == "fit":
            if self.every_n_batches is None:
                return

            print(f"[CustomRayTrainReportCallback] Configuration:")
            print(f"  every_n_batches = {self.every_n_batches}")
            print(f"  accumulate_grad_batches = {trainer.accumulate_grad_batches}")
            print(
                f"  Will checkpoint at session batch indices: {self.every_n_batches}, {self.every_n_batches * 2}, {self.every_n_batches * 3}, ..."
            )
            print(f"  Note: Session batch counter resets on each training start/resume")

    def _save_and_report(
        self,
        trainer: L.Trainer,
        checkpoint_id: str | None = None,
        metrics_only: bool = False,
    ) -> None:
        """Save checkpoint and report to Ray Train.

        Args:
            trainer: The Lightning trainer instance.
            checkpoint_id: The ID of the checkpoint. If None, a random UUID will be generated.
            metrics_only: If True, only report metrics to Ray Train.
        """
        if checkpoint_id is None:
            checkpoint_id = f"step_{trainer.global_step}_{str(uuid.uuid4())[:8]}"

        # Update last checkpointed step to prevent duplicates
        self._last_checkpointed_step = trainer.global_step

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        if not metrics_only:
            # Save checkpoint to local
            tmpdir = Path(self.tmpdir_prefix, checkpoint_id).as_posix()
            os.makedirs(tmpdir, exist_ok=True)
            ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
            trainer.save_checkpoint(ckpt_path, weights_only=False)

            # Report to train session
            checkpoint = Checkpoint.from_directory(tmpdir)
        else:
            tmpdir = None
            checkpoint = None

        ray.train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            if tmpdir is not None:
                shutil.rmtree(tmpdir)

        if not metrics_only:
            self._current_checkpoint_id += 1

    def on_train_start(self, trainer, pl_module) -> None:
        """Called at the start of training."""
        # If training is resumed from a checkpoint, trainer.global_step will be > 0
        # Initialize _last_checkpointed_step to prevent immediate duplicate checkpointing
        if trainer.global_step > 0:
            print(
                f"[CustomRayTrainReportCallback] Resuming from step {trainer.global_step}"
            )
            print(f"[CustomRayTrainReportCallback] Session batch counter reset to 0")
            self._last_checkpointed_step = trainer.global_step
            self.is_first_epoch_since_resumed = True

    def state_dict(self) -> dict:
        """Save callback state to checkpoint."""
        return {
            "_last_checkpointed_step": self._last_checkpointed_step,
            "every_n_batches": self.every_n_batches,
            "every_n_epochs": self.every_n_epochs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore callback state from checkpoint."""
        self._last_checkpointed_step = state_dict.get("_last_checkpointed_step", -1)
        print(
            f"[CustomRayTrainReportCallback] Restored _last_checkpointed_step = {self._last_checkpointed_step}"
        )
        # Note: _session_batch_idx is NOT restored - it resets to 0 on each training session
        # Note: every_n_batches and every_n_epochs are already set in __init__

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Mid-epoch checkpointing based on batch count (since start/resume)."""
        if self.every_n_batches is None:
            return

        # Increment session batch counter
        self._session_batch_idx += 1

        # Debug: Print every 100 batches to show progress
        if self._session_batch_idx % 100 == 0:
            next_checkpoint = (
                (self._session_batch_idx // self.every_n_batches) + 1
            ) * self.every_n_batches
            print(
                f"[CustomRayTrainReportCallback] Session batch: {self._session_batch_idx}, next checkpoint at batch: {next_checkpoint}, current step: {trainer.global_step}"
            )

        # Check if we should checkpoint at this batch
        if self._session_batch_idx % self.every_n_batches == 0:
            step = trainer.global_step

            # Check if we should skip this checkpoint (right after resume)
            # Skip only if we haven't checkpointed anything new since resume
            if (
                self.is_first_epoch_since_resumed
                and step == self._last_checkpointed_step
            ):
                print(
                    f"[CustomRayTrainReportCallback] Skipping checkpoint at session batch {self._session_batch_idx}, step {step} (just resumed from this step)"
                )
                return

            # Clear the resume flag after first new checkpoint
            if (
                self.is_first_epoch_since_resumed
                and step > self._last_checkpointed_step
            ):
                print(
                    f"[CustomRayTrainReportCallback] Resuming normal checkpointing from session batch {self._session_batch_idx}, step {step}"
                )
                self.is_first_epoch_since_resumed = False

            print(
                f"[CustomRayTrainReportCallback] Saving checkpoint at session batch {self._session_batch_idx}, step {step}"
            )
            self._save_and_report(
                trainer,
                checkpoint_id=f"batch_{self._session_batch_idx}_step_{step}",
                metrics_only=False,
            )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Called at the end of each training epoch."""
        if self.every_n_epochs is None:
            return

        # Check if we already checkpointed at this step (e.g., from on_train_batch_end)
        if trainer.global_step == self._last_checkpointed_step:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0:
            self._save_and_report(
                trainer,
                checkpoint_id=f"epoch_{epoch}",
                metrics_only=False,
            )
            self.is_first_epoch_since_resumed = False


class TrainLossLoggingCallback(L.Callback):
    """A callback that logs train loss to MLFlow at custom frequencies.

    Args:
        every_n_steps: Log train loss every N training steps. If None, disabled.
        verbose: If True, print metrics to console when logging. Defaults to False.
    """

    def __init__(
        self, every_n_batches: Optional[int] = None, verbose: bool = False
    ) -> None:
        """
        Args:
            every_n_batches: Log train loss every N training (batch) steps. If None, disabled.
            verbose: If True, print metrics to console when logging. Defaults to False.
        """
        super().__init__()
        self.every_n_batches = every_n_batches
        self.verbose = verbose

        self.every_n_steps = self.every_n_batches

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if stage == "fit":
            if self.every_n_batches is None:
                return

            if (
                trainer.accumulate_grad_batches is not None
                and trainer.accumulate_grad_batches > 1
            ):
                if self.every_n_batches < trainer.accumulate_grad_batches:
                    warnings.warn(
                        f"every_n_batches {self.every_n_batches} is less than accumulate_grad_batches {trainer.accumulate_grad_batches}, setting every_n_steps to 1"
                    )
                    self.every_n_steps = 1
                else:
                    self.every_n_steps = (
                        self.every_n_batches // trainer.accumulate_grad_batches
                    )
            else:
                self.every_n_steps = self.every_n_batches

            print(f"[TrainLossLoggingCallback] Configuration:")
            print(f"  every_n_batches = {self.every_n_batches}")
            print(f"  accumulate_grad_batches = {trainer.accumulate_grad_batches}")
            print(f"  every_n_steps (optimizer steps) = {self.every_n_steps}")
            print(
                f"  Will log at steps: {self.every_n_steps}, {self.every_n_steps * 2}, {self.every_n_steps * 3}, ..."
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Called at the end of each training batch.

        Note: This is called AFTER training_step but AFTER optimizer step only when
        accumulate_grad_batches condition is met.
        """
        if self.every_n_steps is None:
            print("[TrainLossLoggingCallback] Skipping - every_n_steps is None")
            return

        if trainer.global_rank != 0:
            return

        step = trainer.global_step

        # Debug: Check conditions
        if step <= 10:  # Only debug first 10 steps to avoid spam
            print(
                f"[TrainLossLoggingCallback] Debug at batch_idx={batch_idx}, step={step}:"
            )
            print(f"  every_n_steps: {self.every_n_steps}")
            print(f"  step > 0: {step > 0}")
            print(
                f"  step % every_n_steps: {step % self.every_n_steps if self.every_n_steps else 'N/A'}"
            )
            print(f"  should_log: {step > 0 and step % self.every_n_steps == 0}")
            print(f"  callback_metrics keys: {list(trainer.callback_metrics.keys())}")
            if "train_loss" in trainer.callback_metrics:
                print(f"  train_loss value: {trainer.callback_metrics['train_loss']}")

        if step > 0 and step % self.every_n_steps == 0:
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
                    print(
                        f"[TrainLossLoggingCallback] Step {step}: train_loss_custom = {train_loss_value:.4f}"
                    )
            else:
                print(
                    f"[TrainLossLoggingCallback] Warning: 'train_loss' not found in callback_metrics at step {step}"
                )
                print(f"  Available metrics: {list(trainer.callback_metrics.keys())}")

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
