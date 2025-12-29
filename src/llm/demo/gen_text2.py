import tiktoken

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.gpt2.models import GPTModel
from llm.utils import (
    generate,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


def main():
    start = "Every effort moves you"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = text_to_token_ids(start, tokenizer)
    print(f"{encoded=}")

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    token_ids = generate_text_simple(
        model,
        encoded,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    output_text = token_ids_to_text(token_ids, tokenizer)
    print(output_text)

    # with temperature
    token_ids = generate(
        model,
        encoded,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=1.4,
        top_k=25,
    )
    output_text = token_ids_to_text(token_ids, tokenizer)
    print(output_text)


if __name__ == "__main__":
    main()
