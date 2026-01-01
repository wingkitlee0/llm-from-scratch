import argparse
import os
from typing import Any

import lightning as L
import mlflow
import ray
import ray.data
import ray.train
import torch
from lightning.pytorch.loggers import MLFlowLogger
from ray.train import ScalingConfig
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
)
from llm.pretrain.dataset import tokenize_batch
from llm.pretrain.module import GPTLightningModule

DEFAULT_DATA_PATH = "./data/en-parquet"


def train_func(config: dict[str, Any]):
    train_data = ray.train.get_dataset_shard("train")
    val_data = ray.train.get_dataset_shard("val")

    assert train_data is not None
    assert val_data is not None

    train_loader = train_data.iter_torch_batches(
        batch_size=config["train_batch_size"],
        dtypes={"input_ids": torch.long, "labels": torch.long},
        drop_last=True,
        prefetch_batches=2,
    )
    val_loader = val_data.iter_torch_batches(
        batch_size=config["val_batch_size"],
        dtypes={"input_ids": torch.long, "labels": torch.long},
        drop_last=True,
        prefetch_batches=2,
    )

    # 5. Model & Trainer
    model = GPTLightningModule(config["model_config"])

    trainer = L.Trainer(
        strategy=RayDDPStrategy(),
        max_epochs=1,
        accelerator="auto",
        devices=1,
        log_every_n_steps=100,
        plugins=RayLightningEnvironment(),
        logger=MLFlowLogger(
            experiment_name="pretrain",
            run_id=config["mlflow_run_id"],
        ),
        callbacks=[
            CustomRayTrainReportCallback(every_n_steps=1000),
            TrainLossLoggingCallback(every_n_steps=50),
        ],
        val_check_interval=100,  # batches
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    trainer = prepare_trainer(trainer)

    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


def main(data_path: str, model_name: str):
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
    )

    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.enable_operator_progress_bars = False

    train_ds = ray.data.read_parquet(
        os.path.join(data_path, "is_train=1"), columns=["text"]
    ).map_batches(
        tokenize_batch,  # type: ignore
        batch_size=2048,
        batch_format="numpy",
    )

    val_ds = (
        ray.data.read_parquet(os.path.join(data_path, "is_train=0"), columns=["text"])
        .map_batches(
            tokenize_batch,  # type: ignore
            batch_size=2048,
            batch_format="numpy",
        )
        .materialize()
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(
        log_system_metrics=True,
    ) as run:
        trainer = TorchTrainer(
            train_func,
            train_loop_config={
                "model_config": model_config,
                "train_batch_size": 16,
                "val_batch_size": 32,
                "mlflow_run_id": run.info.run_id,
            },
            datasets={
                "train": train_ds,
                "val": val_ds,
            },
            scaling_config=ScalingConfig(
                num_workers=1,
                use_gpu=True,
                resources_per_worker={
                    "GPU": 1,
                    "CPU": 4,
                },
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
    args = parser.parse_args()
    main(
        data_path=os.path.abspath(args.data_path),
        model_name=args.model,
    )
