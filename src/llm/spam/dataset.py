from typing import Optional

import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset

from llm.spam.padding import Padding


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer: tiktoken.Encoding,
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
