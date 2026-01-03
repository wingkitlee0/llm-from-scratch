import argparse
import logging
import os
from typing import Any

import lightning as L
import mlflow
import ray
import ray.data
import ray.train
import torch
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer

from llm.gpt2.pretrained.configs import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG_KEYS,
)
from llm.gpt2.pretrained.utils import get_gpt2_model_config_by_name
from llm.pretrain.callbacks import (
    CustomRayTrainReportCallback,
    IntervalConfig,
    TimingDebugCallback,
    TrainLossLoggingCallback,
)
from llm.pretrain.dataset import tokenize_batch
from llm.pretrain.module import GPTLightningModule

DEFAULT_DATA_PATH = "./data/en-parquet"

logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)


def train_func(config: dict[str, Any]):
    torch.set_float32_matmul_precision("high")
    logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(
        logging.ERROR
    )

    train_data = ray.train.get_dataset_shard("train")
    val_data = ray.train.get_dataset_shard("val")

    assert train_data is not None
    assert val_data is not None

    train_loader = train_data.iter_torch_batches(
        batch_size=config["train_batch_size"],
        dtypes={"input_ids": torch.long, "labels": torch.long},
        drop_last=True,
        prefetch_batches=1,
    )
    val_loader = val_data.iter_torch_batches(
        batch_size=config["val_batch_size"],
        dtypes={"input_ids": torch.long, "labels": torch.long},
        drop_last=True,
        prefetch_batches=1,
    )

    # 5. Model & Trainer
    model = GPTLightningModule.create(config["model_config"])

    trainer = L.Trainer(
        strategy=RayDDPStrategy(),
        max_epochs=1,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        plugins=RayLightningEnvironment(),
        precision=config["precision"],
        logger=[
            MLFlowLogger(
                experiment_name="pretrain",
                run_id=config["mlflow_run_id"],
            ),
            CSVLogger(
                save_dir="logs/csv",
                flush_logs_every_n_steps=1,
            ),
        ],
        callbacks=[
            TimingDebugCallback(),
            CustomRayTrainReportCallback(
                interval=IntervalConfig(every_n_batches=48 * 8)
            ),
            TrainLossLoggingCallback(
                interval=IntervalConfig(every_n_batches=48),
                verbose=True,
            ),
        ],
        accumulate_grad_batches=48,  # 12 * 48 = 576 batches
        limit_val_batches=50,
        val_check_interval=48 * 8,  # batches
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    trainer = prepare_trainer(trainer)

    print("Starting training...")

    # Check for checkpoint from either:
    # 1. Manual restore via train_loop_config["restore_checkpoint"] (when --new-run is used)
    # 2. Automatic Ray Train checkpoint from ray.train.get_checkpoint() (when resuming same experiment)
    checkpoint = config.get("restore_checkpoint") or ray.train.get_checkpoint()

    if checkpoint:
        print("Loading checkpoint from Ray Train...")
        with checkpoint.as_directory() as checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")
            print(f"Checkpoint path: {ckpt_path}")
            # Lightning will restore model, optimizer, epoch, and global_step
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path,
            )
    else:
        print("Starting training from scratch...")
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


def find_latest_checkpoint_manually(restore_path: str) -> str:
    # Find the latest checkpoint in the source directory
    checkpoint_dirs = [
        d for d in os.listdir(restore_path)
        if d.startswith("checkpoint_") and os.path.isdir(os.path.join(restore_path, d))
    ]

    if checkpoint_dirs:
        latest_checkpoint = sorted(checkpoint_dirs)[-1]
        return os.path.join(restore_path, latest_checkpoint)

    raise ValueError(f"No checkpoints found in: {restore_path}")


def main(data_path: str, model_name: str, restore_path: str | None = None, new_run: bool = False):
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_config = get_gpt2_model_config_by_name(model_name=model_name)

    mlflow_tracking_uri = f"sqlite:///{os.path.abspath('mlflow.db')}"

    # 1. Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "py_modules": ["."],
            "excludes": ["models", "gpt2", ".git", ".venv", "data", ".ruff_cache"],
            "env_vars": {
                "MLFLOW_TRACKING_URI": mlflow_tracking_uri,
            },
        },
        object_store_memory=50 * 1024 * 1024 * 1024,  # 50GB
    )

    ctx = ray.data.DataContext.get_current()
    ctx.target_max_block_size = 20 * 1024 * 1024  # 20MB
    # ctx.enable_rich_progress_bars = False
    # ctx.enable_operator_progress_bars = False

    train_ds = ray.data.read_parquet(
        os.path.join(data_path, "is_train=1"),
        columns=["text"],
        shuffle="files",
    ).map_batches(
        tokenize_batch,  # type: ignore
        batch_size=8,
        batch_format="numpy",
        zero_copy_batch=True,
        compute=ray.data.TaskPoolStrategy(size=6),  # max 10 tasks
    )

    val_ds = (
        ray.data.read_parquet(
            os.path.join(data_path, "is_train=0"),
            columns=["text"],
            shuffle="files",
        )
        .map_batches(
            tokenize_batch,  # type: ignore
            batch_size=32,
            batch_format="numpy",
            zero_copy_batch=True,
        )
        .limit(50 * 128)
        .materialize()
    )

    # print(f"{val_ds.count()=}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(
        log_system_metrics=True,
    ) as run:
        train_loop_config = {
            "model_config": model_config,
            "train_batch_size": 12,
            "val_batch_size": 64,
            "mlflow_run_id": run.info.run_id,
            "precision": "bf16-mixed",
        }

        # Ray Train v2: Configure RunConfig for checkpoint restoration
        if restore_path:
            restore_path = os.path.abspath(restore_path)
            storage_path = os.path.dirname(restore_path)
            experiment_name = os.path.basename(restore_path)

            if new_run:
                # Create a new run but load checkpoint from the old experiment
                print(f"Creating new run while loading checkpoint from: {restore_path}")
                print(f"  Checkpoint source: {restore_path}")
                print(f"  New run will be created in: {storage_path}/")

                checkpoint_path = find_latest_checkpoint_manually(restore_path)
                print(f"  Loading checkpoint: {checkpoint_path}")
                train_loop_config["restore_checkpoint"] = Checkpoint.from_directory(checkpoint_path)

                # Create new experiment (don't set name, let Ray Train generate one)
                run_config = RunConfig(
                    storage_path=storage_path,
                    failure_config=FailureConfig(3),
                )
            else:
                # Resume in the same experiment directory
                print(f"Resuming training from: {restore_path}")
                print(f"  storage_path: {storage_path}")
                print(f"  experiment_name: {experiment_name}")
                print("Ray Train will automatically load the latest checkpoint")

                run_config = RunConfig(
                    storage_path=storage_path,
                    name=experiment_name,
                    failure_config=FailureConfig(3),
                )
        else:
            # New training - use default storage (./ray_results)
            run_config = RunConfig(
                failure_config=FailureConfig(3),
            )

        trainer = TorchTrainer(
            train_func,
            train_loop_config=train_loop_config,
            datasets={
                "train": train_ds,
                "val": val_ds,
            },
            scaling_config=ScalingConfig(
                num_workers=1,
                use_gpu=True,
                resources_per_worker={
                    "GPU": 1,
                    "CPU": 6,
                },
            ),
            run_config=run_config,
        )

        result = trainer.fit()

        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Data path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=MODEL_CONFIG_KEYS,
        help=f"Model to use. Available models: {', '.join(MODEL_CONFIG_KEYS)}",
    )
    parser.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Path to Ray Train checkpoint directory to restore from",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Create a new experiment run while loading checkpoint from restore-path. "
             "Useful for continuing training as a separate experiment.",
    )
    args = parser.parse_args()
    main(
        data_path=os.path.abspath(args.data_path),
        model_name=args.model,
        restore_path=args.restore_path,
        new_run=args.new_run,
    )
