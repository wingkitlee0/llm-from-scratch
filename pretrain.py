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
from ray.train import FailureConfig, RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer

from llm.gpt2.pretrained.configs import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG_KEYS,
)
from llm.gpt2.pretrained.utils import get_gpt2_model_config_by_name
from llm.pretrain.callbacks import (
    CustomRayTrainReportCallback,
    TrainLossLoggingCallback,
    TimingDebugCallback,
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
    model = GPTLightningModule(config["model_config"])

    trainer = L.Trainer(
        strategy=RayDDPStrategy(),
        max_epochs=1,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        plugins=RayLightningEnvironment(),
        logger=[
            MLFlowLogger(
                experiment_name="pretrain",
                run_id=config["mlflow_run_id"],
            ),
            CSVLogger(
                save_dir="logs/csv",
                flush_logs_every_n_steps=10,
            ),
        ],
        callbacks=[
            TimingDebugCallback(),
            CustomRayTrainReportCallback(every_n_batches=512),
            TrainLossLoggingCallback(
                every_n_batches=48,
                verbose=True,
            ),
        ],
        accumulate_grad_batches=48,  # 48 * 8 = 384 batches
        limit_val_batches=50,
        val_check_interval=50,  # batches
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    trainer = prepare_trainer(trainer)

    print("Starting training...")

    # Check for checkpoint from either:
    # 1. Manual restore via train_loop_config["restore_checkpoint"]
    # 2. Automatic Ray Train checkpoint from ray.train.get_checkpoint()
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


def main(data_path: str, model_name: str, restore_path: str | None = None):
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
        # Ray Train v2: Pass checkpoint via train_loop_config for manual restoration
        train_loop_config = {
            "model_config": model_config,
            "train_batch_size": 12,
            "val_batch_size": 64,
            "mlflow_run_id": run.info.run_id,
        }

        if restore_path:
            # Find the latest checkpoint in the restore_path directory
            print(f"Looking for checkpoints in: {restore_path}")
            checkpoint_dirs = [
                d
                for d in os.listdir(restore_path)
                if d.startswith("checkpoint_")
                and os.path.isdir(os.path.join(restore_path, d))
            ]
            if checkpoint_dirs:
                # Sort by name (which includes timestamp) to get the latest
                latest_checkpoint = sorted(checkpoint_dirs)[-1]
                checkpoint_path = os.path.join(restore_path, latest_checkpoint)
                print(f"Found checkpoint: {checkpoint_path}")

                from ray.train import Checkpoint

                train_loop_config["restore_checkpoint"] = Checkpoint.from_directory(
                    checkpoint_path
                )
            else:
                print(f"Warning: No checkpoints found in {restore_path}")

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
            run_config=RunConfig(
                failure_config=FailureConfig(3),
            ),
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
    args = parser.parse_args()
    main(
        data_path=os.path.abspath(args.data_path),
        model_name=args.model,
        restore_path=args.restore_path,
    )
