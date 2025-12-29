import argparse

import torch

from llm.common import get_tokenizer
from llm.gpt2.pretrained import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG_KEYS,
    get_gpt2_model_with_weights,
)
from llm.utils import generate, text_to_token_ids, token_ids_to_text


def main(model_name: str = DEFAULT_MODEL_NAME):
    model, model_config = get_gpt2_model_with_weights(model_name=model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # torch.manual_seed(123)
    tokenizer = get_tokenizer()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Tell me a story about a cat.", tokenizer).to(device),
        max_new_tokens=200,
        context_size=model_config["context_length"],
        top_k=50,
        temperature=0.8,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and run GPT-2 models from OpenAI checkpoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=MODEL_CONFIG_KEYS,
        help=f"Model to use. Available models: {', '.join(MODEL_CONFIG_KEYS)}",
    )
    args = parser.parse_args()
    main(model_name=args.model)
