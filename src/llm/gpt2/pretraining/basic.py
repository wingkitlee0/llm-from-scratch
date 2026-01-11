import torch

from llm.dataloader import create_dataloader_v1


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )
    return loss


def calc_loss_loader(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    num_batches = num_batches or len(data_loader)
    num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def get_train_and_val_loaders(
    file_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    with open(file_path, "r") as f:
        text_data = f.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Total characters: {total_characters}")
    print(f"Total tokens: {total_tokens}")

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def calc_loss_train_and_val(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
):
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
