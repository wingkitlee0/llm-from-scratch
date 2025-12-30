from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class EvaluateResult:
    train_loss: float
    val_loss: float


def evaluate_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_iter: int,
    calc_loss_loader_fn: Callable[
        [torch.utils.data.DataLoader, torch.nn.Module, torch.device, int], float
    ],
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_fn(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader_fn(val_loader, model, device, eval_iter)

    model.train()
    return EvaluateResult(
        train_loss=train_loss,
        val_loss=val_loss,
    )
