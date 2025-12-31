import os
import time
from typing import Optional, Dict, Any, Iterator
import argparse
import lightning as L
import ray
import ray.data
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from llm.configs.gpt_config import DEFAULT_GPT_CONFIG
from llm.gpt2.models import GPTModel
from llm.gpt2.pretrained.configs import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIGS,
    MODEL_CONFIG_KEYS,
)
from llm.gpt2.pretrained.utils import get_gpt2_model_config_by_name


class GPTLightningModule(L.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        """
        PyTorch Lightning module for GPT-2 pretraining.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = GPTModel(config)
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Ray iter_torch_batches yields a dict with keys from dtypes
        inputs, targets = batch["input_ids"], batch["labels"]

        logits = self.model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Using simple AdamW for demonstration; production scripts usually use cosine schedule
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=4e-4,
            weight_decay=0.1
        )
        return optimizer
