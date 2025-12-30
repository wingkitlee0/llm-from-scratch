# ch5
import torch

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.gpt2.pretraining.basic import (
    calc_loss_train_and_val,
    get_train_and_val_loaders,
)
from llm.utils import get_gpt_model, get_tokenizer


def main():
    tokenizer = get_tokenizer()
    model = get_gpt_model()

    train_loader, val_loader = get_train_and_val_loaders(
        file_path="data/the-verdict.txt",
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        num_workers=0,
    )

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    calc_loss_train_and_val(train_loader, val_loader, model, device)


if __name__ == "__main__":
    main()
