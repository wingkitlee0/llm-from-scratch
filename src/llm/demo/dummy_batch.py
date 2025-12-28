import tiktoken
import torch
from torch.nn.utils.rnn import pad_sequence


def get_dummy_batch() -> torch.Tensor:
    tokenizer = tiktoken.get_encoding("gpt2")
    texts = [
        "Hello, world!",
        "Hello, how are you?",
        "I am fine, thank you!",
        "How are you doing?",
        "I am doing great, thank you!",
        "What is your name?",
        "My name is John Doe.",
        "What is your favorite color?",
        "My favorite color is blue.",
        "What is your favorite food?",
        "My favorite food is pizza.",
    ]

    batch = [torch.tensor(tokenizer.encode(text)) for text in texts]
    batch = pad_sequence(batch, batch_first=True, padding_value=0)

    return batch
