from typing import Any, Dict

import lightning as L
import torch
import torch.nn.functional as F

from llm.gpt2.models import GPTModel, GPTModelV2


class GPTLightningModule(L.LightningModule):
    @classmethod
    def create(cls, config: Dict[str, Any]) -> "GPTLightningModule":
        if config.get("enable_flash_att", False):
            model = GPTModelV2(config)
        else:
            model = GPTModel(config)
        return cls(model=model, config=config)

    def __init__(self, model, config: Dict[str, Any]):
        """
        PyTorch Lightning module for GPT-2 pretraining.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
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

    def validation_step(self, batch, batch_idx):
        # Ray iter_torch_batches yields a dict with keys from dtypes
        inputs, targets = batch["input_ids"], batch["labels"]

        # Debug: print to verify validation is running
        if batch_idx == 0:
            print(
                f"\n[VALIDATION] Step 0 - inputs shape: {inputs.shape}, targets shape: {targets.shape}"
            )

        logits = self.model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        # Using simple AdamW for demonstration; production scripts usually use cosine schedule
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,  # Reduced from 4e-4 for more stable training with small batch
            weight_decay=0.1,
        )
        return optimizer
