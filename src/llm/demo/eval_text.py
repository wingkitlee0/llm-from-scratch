import torch

from llm.utils import (
    get_gpt_model,
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)


def main():
    input_texts = [
        "every effort moves",
        "I really like",
    ]

    tokenizer = get_tokenizer()
    input_token_ids = text_to_token_ids(input_texts, tokenizer)
    print(f"{input_token_ids.shape=}")
    print(f"{input_token_ids=}")

    target_texts = [
        " effort moves you",
        " really like chocolate",
    ]

    target_token_ids = text_to_token_ids(target_texts, tokenizer)
    print(f"{target_token_ids.shape=}")
    print(f"{target_token_ids=}")

    model = get_gpt_model()
    model.eval()

    with torch.no_grad():
        # (batch_size, num_tokens, vocab_size)
        logits = model(input_token_ids)
    # (batch_size, num_tokens, vocab_size) = (2, 3, 50257)
    probs = torch.softmax(logits, dim=-1)
    print(f"{probs.shape=}")
    predicted_token_ids = torch.argmax(probs, dim=-1, keepdim=True)
    print(f"{predicted_token_ids.shape=}")
    print(f"{predicted_token_ids=}")

    predicted_texts = [
        token_ids_to_text(predicted_token_ids[i].flatten(), tokenizer)
        for i in range(predicted_token_ids.shape[0])
    ]
    print(f"{predicted_texts=}")

    text_idx = 0  # first batch
    target_probas_1 = probs[text_idx, [0, 1, 2], target_token_ids[text_idx]]
    print("Text 1: ", target_probas_1)

    text_idx = 1
    target_probas_2 = probs[text_idx, [0, 1, 2], target_token_ids[text_idx]]
    print("Text 2: ", target_probas_2)

    # shape = ( batch_size * num_tokens, ) = (6, )
    log_probs = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print("Log probabilities: ", log_probs)

    average_log_prob = log_probs.mean()
    print("Average log probability: ", average_log_prob)

    neg_average_log_prob = -average_log_prob
    print("Negative average log probability: ", neg_average_log_prob)

    logits_flat = logits.flatten(0, 1)
    targets_flat = target_token_ids.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Loss: ", loss)

    perplexity = torch.exp(loss)
    print("Perplexity: ", perplexity)


if __name__ == "__main__":
    main()
