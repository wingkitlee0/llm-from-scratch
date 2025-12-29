import argparse
import time
from typing import TYPE_CHECKING, Optional

import torch

from llm.common import get_tokenizer
from llm.gpt2.pretrained import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG_KEYS,
    get_gpt2_model_with_weights,
)
from llm.spam.classifier import (
    calc_accuracy_loader,
    calc_loss_loader,
    prepare_gpt2_for_classification,
    train_classifier_simple,
)
from llm.spam.classifier.model_io import save_classifier_model
from llm.spam.dataset import (
    get_dataloaders,
    get_datasets,
)
from llm.utils import (
    generate,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)

if TYPE_CHECKING:
    import tiktoken

DEFAULT_DATA_DIR = "data/sms_spam_collection/20251229180023"


def check_dataloaders(dataloaders: dict[str, torch.utils.data.DataLoader]):
    for input_batch, target_batch in dataloaders["train"]:
        print(f"{input_batch.shape=}")
        print(f"{target_batch.shape=}")
        break

    for key, dataloader in dataloaders.items():
        print(f"{key}: {len(dataloader)}")


def calc_accuracy_all(
    dataloaders: dict[str, torch.utils.data.DataLoader],
    model: torch.nn.Module,
    device: torch.device,
):
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


def calc_loss_all(
    dataloaders: dict[str, torch.utils.data.DataLoader],
    model: torch.nn.Module,
    device: torch.device,
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            dataloaders["train"], model, device, num_batches=10
        )
        val_loss = calc_loss_loader(dataloaders["val"], model, device, num_batches=10)
        test_loss = calc_loss_loader(dataloaders["test"], model, device, num_batches=10)
    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    model.train()


def check_basic_examples(
    model: torch.nn.Module,
    tokenizer: "tiktoken.Encoding",
    device: torch.device,
    model_config: dict,
):
    text_1 = "Tell me a story about a cat."
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


def check_model_parameters(
    model: torch.nn.Module, tokenizer: "tiktoken.Encoding", device: torch.device
):
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


def main(
    data_dir: str,
    model_name: str,
    device: torch.device,
    tokenizer: Optional["tiktoken.Encoding"] = None,
):
    tokenizer = tokenizer or get_tokenizer()

    datasets = get_datasets(
        data_dir=data_dir,
        tokenizer=tokenizer,
    )

    num_workers = 0
    batch_size = 8

    dataloaders = get_dataloaders(datasets, batch_size, num_workers)

    check_dataloaders(dataloaders)

    model, model_config = get_gpt2_model_with_weights(model_name=model_name)
    model.to(device)

    check_basic_examples(model, tokenizer, device, model_config)

    model = prepare_gpt2_for_classification(model, model_config, device)
    model.to(device)
    print(model)

    check_model_parameters(model, tokenizer, device)

    calc_accuracy_all(dataloaders, model, device)

    calc_loss_all(dataloaders, model, device)

    start_time = time.perf_counter()

    # Verify out_head parameters are in optimizer
    all_params = list(model.parameters())
    out_head_params = list(model.out_head.parameters())
    print(f"Total model parameters: {len(all_params)}")
    print(
        f"out_head parameters in model.parameters(): {any(id(p) in [id(ap) for ap in all_params] for p in out_head_params)}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_results = train_classifier_simple(
        model,
        dataloaders["train"],
        dataloaders["val"],
        optimizer,
        device,
        num_epochs,
        eval_freq=50,
        eval_iter=5,
    )

    end_time = time.perf_counter()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    save_classifier_model(
        model,
        model_config,
        datasets["train"].max_length,
        train_results,
        model_name,
    )

    print("After finetuning:")
    calc_accuracy_all(dataloaders, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Data directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=MODEL_CONFIG_KEYS,
        help=f"Model to use. Available models: {', '.join(MODEL_CONFIG_KEYS)}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
    )
    args = parser.parse_args()

    if args.device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device(args.device)

    main(
        data_dir=args.data_dir,
        model_name=args.model,
        device=device,
    )
