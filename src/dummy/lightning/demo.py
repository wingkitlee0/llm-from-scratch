import logging

import ray
import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from ray.data.expressions import lit

from dummy.lightning.callbacks import LoudCallback as LoudCallbackV2

logging.getLogger("ray.train").setLevel("WARNING")
logging.getLogger("ray.data").setLevel("WARNING")


class DummyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x).squeeze(-1)  # Squeeze to [batch_size]

    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["y"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch["data"], batch["y"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f"val_loss_{dataloader_idx}", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def get_dummy_dataloader(
    num_samples=100,
    shape=(10,),
    batch_size=10,
    dtypes=torch.float32,
    y_value=1.0,
):
    return (
        ray.data.range_tensor(num_samples, shape=shape)
        .with_column("y", lit(y_value))
        .iter_torch_batches(batch_size=batch_size, dtypes=dtypes)
    )


def main():
    logging.basicConfig(level="WARNING")

    # ctx = ray.data.DataContext.get_current()
    # ctx.enable_progress_bars = False
    # ctx.use_ray_tqdm = False
    # ctx.enable_operator_progress_bars = False
    # ctx.enable_rich_progress_bars = False

    module = DummyModule()
    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=2,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=True,
        callbacks=[
            # LoudCallback(),
            LoudCallbackV2(),
        ],
    )
    trainer.fit(
        module,
        train_dataloaders=get_dummy_dataloader(),
        val_dataloaders=[
            get_dummy_dataloader(),
            get_dummy_dataloader(),
        ],
    )


if __name__ == "__main__":
    main()
