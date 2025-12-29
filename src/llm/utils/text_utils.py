from typing import Optional

import tiktoken
import torch


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
    device: Optional[torch.device] = None,
):
    """
    Args:
        model: the model to use for text generation
        idx: the vocab indices of the input tokens, shape (batch_size, n_tokens)
        max_new_tokens: the maximum number of new tokens to generate
        context_size: the size of the context
    """
    if device is not None:
        idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            # Output shape: (batch_size, num_tokens, vocab_size)
            logits = model(idx_cond)

        # Only use the last time step
        logits = logits[:, -1, :]  #  (batch_size, vocab_size)
        # When applying softmax to the last vocab dimension,
        # the shape does not change.
        probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)
        # We use argmax to get the most probable vocab index (one number per batch)
        # Hence, the shape is (batch_size, 1) as we keep the last dimension.
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        # Concatenate the new token to the input tokens
        # shape: (batch_size, n_tokens + 1)
        # Note that both idx and idx_next are 2D.
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


def text_to_token_ids(
    text: str | list[str], tokenizer: tiktoken.Encoding
) -> torch.Tensor:
    def _single_encode(text: str) -> torch.Tensor:
        encoded = tokenizer.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )
        return torch.tensor(encoded).unsqueeze(0)

    if isinstance(text, str):
        return _single_encode(text)
    elif isinstance(text, list):
        return torch.cat([_single_encode(t) for t in text])
    else:
        raise ValueError(f"Unsupported type: {type(text)}")


def token_ids_to_text(
    token_ids: torch.Tensor,
    tokenizer: tiktoken.Encoding,
) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_token_id=None,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_token_id:
            break

        idx = torch.cat([idx, idx_next], dim=1)

    return idx
