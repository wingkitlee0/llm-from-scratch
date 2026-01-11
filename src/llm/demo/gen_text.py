import tiktoken
import torch

from llm.configs.gpt_config import DEFAULT_GPT_CONFIG
from llm.gpt2.models import GPTModel
from llm.utils import generate_text_simple


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)

    print(f"{encoded=}")

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"{encoded_tensor.shape=}")

    model = GPTModel(DEFAULT_GPT_CONFIG)
    model.eval()

    output = generate_text_simple(
        model,
        encoded_tensor,
        max_new_tokens=6,
        context_size=DEFAULT_GPT_CONFIG["context_length"],
    )
    print(f"{output.shape=}")
    print(output)
    decoded = tokenizer.decode(output.squeeze(0).tolist())
    print(decoded)


if __name__ == "__main__":
    main()
