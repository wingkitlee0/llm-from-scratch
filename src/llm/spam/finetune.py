import argparse
import os
from typing import Optional

import torch

from llm.common import get_tokenizer
from llm.gpt2.pretrained import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG_KEYS,
    get_gpt2_model_with_weights,
)
from llm.spam.dataset import (
    SpamDataset,
)
from llm.spam.padding import Padding
from llm.utils import (
    generate,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


def get_datasets(data_dir) -> dict[str, SpamDataset]:
    tokenizer = get_tokenizer()

    train_dataset = SpamDataset(
        csv_file=os.path.join(data_dir, "train.csv"),
        tokenizer=tokenizer,
        max_length=None,
    )
    val_dataset = SpamDataset(
        csv_file=os.path.join(data_dir, "val.csv"),
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )
    test_dataset = SpamDataset(
        csv_file=os.path.join(data_dir, "test.csv"),
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def calc_accuracy_loader(
    data_loader, model, device, num_batches=None, padding: Optional[Padding] = None
):
    padding = padding or Padding()

    model.eval()

    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                all_logits = model(input_batch)
                logits = padding.extract_logits_at_last_non_padding(
                    all_logits, input_batch
                )
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    model.train()
    return correct_predictions / num_examples


def calc_loss_batch(
    input_batch, target_batch, model, device, padding: Optional[Padding] = None
):
    padding = padding or Padding()
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    all_logits = model(input_batch)
    logits = padding.extract_logits_at_last_non_padding(all_logits, input_batch)

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def main(model_name: str):
    datasets = get_datasets(data_dir="data/sms_spam_collection/20251229180023")
    # Get max_length from training dataset (used for padding)
    max_length = datasets["train"].max_length

    num_workers = 0
    batch_size = 8
    torch.manual_seed(1234)

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
    }

    for input_batch, target_batch in dataloaders["train"]:
        print(f"{input_batch.shape=}")
        print(f"{target_batch.shape=}")
        break

    for key, dataloader in dataloaders.items():
        print(f"{key}: {len(dataloader)}")

    model, model_config = get_gpt2_model_with_weights(model_name=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text_1 = "Tell me a story about a cat."
    tokenizer = get_tokenizer()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer).to(device),
        max_new_tokens=200,
        context_size=model_config["context_length"],
        temperature=0.7,
        top_k=50,
    )
    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer).to(device),
        max_new_tokens=23,
        context_size=model_config["context_length"],
    )
    print(token_ids_to_text(token_ids, tokenizer))

    print(model)

    # Add a classification layer
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=model_config["emb_dim"],
        out_features=num_classes,
        device=device,
    )

    # Ensure out_head parameters are trainable
    for param in model.out_head.parameters():
        param.requires_grad = True

    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    # Verify out_head is registered and has trainable parameters
    out_head_params = list(model.out_head.parameters())
    print(f"out_head parameters: {len(out_head_params)}")
    print(f"out_head requires_grad: {all(p.requires_grad for p in out_head_params)}")
    print(model)

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)

    with torch.no_grad():
        outputs = model(inputs.to(device))

    print(f"{outputs.shape=}")
    print(f"{outputs=}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(
        dataloaders["train"], model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
        dataloaders["val"], model, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
        dataloaders["test"], model, device, num_batches=10
    )
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    with torch.no_grad():
        train_loss = calc_loss_loader(
            dataloaders["train"], model, device, num_batches=10
        )
        val_loss = calc_loss_loader(dataloaders["val"], model, device, num_batches=10)
        test_loss = calc_loss_loader(dataloaders["test"], model, device, num_batches=10)
    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    import time

    start_time = time.perf_counter()
    torch.manual_seed(123)

    # Verify out_head parameters are in optimizer
    all_params = list(model.parameters())
    out_head_params = list(model.out_head.parameters())
    print(f"Total model parameters: {len(all_params)}")
    print(
        f"out_head parameters in model.parameters(): {any(id(p) in [id(ap) for ap in all_params] for p in out_head_params)}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model,
            dataloaders["train"],
            dataloaders["val"],
            optimizer,
            device,
            num_epochs,
            eval_freq=50,
            eval_iter=5,
        )
    )
    end_time = time.perf_counter()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save the model
    model_dir = "models/spam_classifier"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_finetuned.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "max_length": max_length,  # Save max_length for inference
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    print("After finetuning:")
    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(
        dataloaders["train"], model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
        dataloaders["val"], model, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
        dataloaders["test"], model, device, num_batches=10
    )
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=MODEL_CONFIG_KEYS,
        help=f"Model to use. Available models: {', '.join(MODEL_CONFIG_KEYS)}",
    )
    args = parser.parse_args()
    main(model_name=args.model)
