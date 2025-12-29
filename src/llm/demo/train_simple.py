import tiktoken
import torch

from llm.common import get_gpt_model, get_tokenizer
from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.demo.load_data import (
    calc_loss_batch,
    calc_loss_loader,
    get_train_and_val_loaders,
)
from llm.gpt2.models import GPTModel
from llm.utils import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


def evaluate_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_iter: int,
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model,
            encoded,
            max_new_tokens=6,
            context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model: GPTModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                tokens_seen = 0
                global_step = 0

                print(
                    f"Epoch {epoch + 1}, Step {global_step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_gpt_model()
    model.to(device)
    tokenizer = get_tokenizer()

    train_loader, val_loader = get_train_and_val_loaders(
        file_path="data/the-verdict.txt",
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1,
    )

    num_epochs = 10

    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
