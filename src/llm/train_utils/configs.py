from typing import Any

from pydantic import BaseModel

DEFAULT_TRAIN_LOOP_CONFIG_FOR_DGX_SPARK = {
    "train_batch_size": 32,
    "val_batch_size": 64,
    "precision": "bf16-mixed",
    "max_epochs": 1,
}

DEFAULT_TRAIN_LOOP_CONFIG_FOR_SMOKETEST = {
    "train_batch_size": 1,
    "val_batch_size": 1,
    "precision": "32",
    "max_epochs": 10,
}


class TrainLoopConfig(BaseModel):
    gpt2_config: dict[str, Any]
    mlflow_run_id: str
    train_batch_size: int
    val_batch_size: int
    precision: str
    max_epochs: int
    restore_checkpoint_path: str | None = None

    @classmethod
    def create_for_dgx_spark(
        cls,
        gpt2_config: dict[str, Any],
        mlflow_run_id: str,
        **kwargs,
    ) -> "TrainLoopConfig":
        return cls.create(
            gpt2_config=gpt2_config,
            mlflow_run_id=mlflow_run_id,
            default_configs=DEFAULT_TRAIN_LOOP_CONFIG_FOR_DGX_SPARK,
            **kwargs,
        )

    @classmethod
    def create_for_smoketest(
        cls, gpt2_config: dict[str, Any], mlflow_run_id: str, **kwargs
    ) -> "TrainLoopConfig":
        return cls.create(
            gpt2_config=gpt2_config,
            mlflow_run_id=mlflow_run_id,
            default_configs=DEFAULT_TRAIN_LOOP_CONFIG_FOR_SMOKETEST,
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        gpt2_config: dict[str, Any],
        mlflow_run_id: str,
        default_configs: dict[str, Any],
        **kwargs,
    ) -> "TrainLoopConfig":
        return cls(
            gpt2_config=gpt2_config,
            mlflow_run_id=mlflow_run_id,
            **{**default_configs, **kwargs},
        )
