import tiktoken
import torch


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
):
    """
    Args:
        model: the model to use for text generation
        idx: the vocab indices of the input tokens, shape (batch_size, n_tokens)
        max_new_tokens: the maximum number of new tokens to generate
        context_size: the size of the context
    """
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
