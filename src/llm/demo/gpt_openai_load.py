import argparse
from importlib import resources

import torch
import yaml

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.demo.gpt_download import download_and_load_gpt2
from llm.demo.gpt_load import load_weights_into_gpt
from llm.demo.text_utils import generate, text_to_token_ids, token_ids_to_text
from llm.gpt_models import GPTModel
from llm.top_level import get_tokenizer

DEFAULT_MODEL_NAME = "gpt2-small"

# Load model configs from YAML file
with resources.files("llm.configs").joinpath("model_configs.yaml").open("r") as f:
    MODEL_CONFIGS = yaml.safe_load(f)


def get_model_download_size(model_name: str) -> str:
    """Get the download size (e.g., '124M') from model config."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model name: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_name]["size"]


def get_gpt_model_openai(model_name: str = DEFAULT_MODEL_NAME) -> tuple[GPTModel, dict]:
    new_config = GPT_CONFIG_124M.copy()
    new_config.update(MODEL_CONFIGS[model_name]["config"])
    new_config.update({"context_length": 1024})
    new_config.update({"qkv_bias": True})

    return GPTModel(new_config), new_config


def main(model_name: str = DEFAULT_MODEL_NAME):
    model_size = get_model_download_size(model_name)
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model, model_config = get_gpt_model_openai(model_name=model_name)
    load_weights_into_gpt(model, params)
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
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Model to use. Available models: {', '.join(MODEL_CONFIGS.keys())}",
    )
    args = parser.parse_args()
    main(model_name=args.model)
