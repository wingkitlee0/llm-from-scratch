import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import lightning as L
import ray
import ray.train
from ray.train import Checkpoint


class CustomRayTrainReportCallback(L.Callback):
    """A callback that reports checkpoints to Ray at custom frequencies.

    Args:
        every_n_steps: Save checkpoint every N training steps. If None, disabled.
        every_n_epochs: Save checkpoint every N epochs. If None, disabled.
            Defaults to 1 for backward compatibility.
        save_on_train_epoch_end: Whether to save at the end of each epoch.
            Deprecated in favor of every_n_epochs.
    """

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(
        self,
        every_n_steps: Optional[int] = None,
        every_n_epochs: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self._last_checkpointed_step = -1  # Track last checkpoint to avoid duplicates

        job_id = ray.get_runtime_context().get_job_id()
        experiment_name = ray.train.get_context().get_experiment_name()
        self.local_rank = ray.train.get_context().get_local_rank()

        self.tmpdir_prefix = Path(
            tempfile.gettempdir(),
            f"lightning_checkpoints-job_id={job_id}-name={experiment_name}",
        ).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

    def _save_and_report(self, trainer, checkpoint_id: str) -> None:
        """Save checkpoint and report to Ray Train."""
        # Update last checkpointed step to prevent duplicates
        self._last_checkpointed_step = trainer.global_step

        tmpdir = Path(self.tmpdir_prefix, checkpoint_id).as_posix()
        os.makedirs(tmpdir, exist_ok=True)

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Save checkpoint to local
        ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = Checkpoint.from_directory(tmpdir)
        ray.train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Called at the end of each training batch."""
        if self.every_n_steps is None:
            return

        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            self._save_and_report(trainer, checkpoint_id=f"step_{step}")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Called at the end of each training epoch."""
        if self.every_n_epochs is None:
            return

        # Check if we already checkpointed at this step (e.g., from on_train_batch_end)
        if trainer.global_step == self._last_checkpointed_step:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0:
            self._save_and_report(trainer, checkpoint_id=f"epoch_{epoch}")


class TrainLossLoggingCallback(L.Callback):
    """A callback that logs train loss to MLFlow at custom frequencies.

    Args:
        every_n_steps: Log train loss every N training steps. If None, disabled.
        verbose: If True, print metrics to console when logging. Defaults to False.
    """

    def __init__(
        self, every_n_steps: Optional[int] = None, verbose: bool = False
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.verbose = verbose

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Called at the end of each training batch."""
        if self.every_n_steps is None:
            return

        if trainer.global_rank != 0:
            return

        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            # Get train_loss from callback_metrics
            if "train_loss" in trainer.callback_metrics:
                train_loss = trainer.callback_metrics["train_loss"]
                # Convert to float for printing if it's a tensor
                train_loss_value = (
                    train_loss.item()
                    if hasattr(train_loss, "item")
                    else float(train_loss)
                )

                # Log using pl_module.log() which automatically routes to all loggers (MLFlow, etc.)
                # Use a custom metric name to distinguish from default train_loss logging
                pl_module.log(
                    "train_loss_custom", train_loss, on_step=True, logger=True
                )

                # Print metrics if verbose is enabled
                if self.verbose:
                    print(
                        f"[TrainLossLoggingCallback] Step {step}: train_loss_custom = {train_loss_value:.4f}"
                    )
