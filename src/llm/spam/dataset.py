import os
from typing import TYPE_CHECKING, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from llm.spam.padding import Padding

if TYPE_CHECKING:
    import tiktoken


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer: "tiktoken.Encoding",
        max_length: int | None = None,
        padding: Optional[Padding] = None,
    ):
        self.df = pd.read_csv(csv_file)
        self.padding = padding or Padding()
        self.encoded_texts = [tokenizer.encode(text) for text in self.df["text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [text[:max_length] for text in self.encoded_texts]

        self.encoded_texts = self.padding.pad_sequences(
            self.encoded_texts, self.max_length
        )

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.df.iloc[idx]["label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def get_datasets(
    data_dir: str, tokenizer: "tiktoken.Encoding"
) -> dict[str, SpamDataset]:
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


def get_dataloaders(
    datasets: dict[str, SpamDataset], batch_size: int, num_workers: int
) -> dict[str, torch.utils.data.DataLoader]:
    return {
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
