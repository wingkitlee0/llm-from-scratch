import tiktoken
import torch


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"},
    )
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(
    token_ids: torch.Tensor,
    tokenizer: tiktoken.Encoding,
) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
