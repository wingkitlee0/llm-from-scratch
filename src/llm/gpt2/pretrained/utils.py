from llm.configs.gpt_config import DEFAULT_GPT_CONFIG
from llm.gpt2.models import GPTModel

from .configs import DEFAULT_MODEL_NAME, MODEL_CONFIGS
from .download import download_and_load_gpt2
from .load import load_weights_into_gpt2


def get_model_download_size(model_name: str) -> str:
    """Get the download size (e.g., '124M') from model config."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model name: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_name]["size"]


def get_pretrained_gpt2_model(model_name: str) -> tuple[GPTModel, dict]:
    new_config = DEFAULT_GPT_CONFIG.copy()
    new_config.update(MODEL_CONFIGS[model_name]["config"])
    new_config.update({"context_length": 1024})
    new_config.update({"qkv_bias": True})

    return GPTModel(new_config), new_config


def get_gpt2_model_with_weights(
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[GPTModel, dict]:
    model_size = get_model_download_size(model_name)
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model, model_config = get_pretrained_gpt2_model(model_name=model_name)
    load_weights_into_gpt2(model, params)
    return model, model_config


def get_gpt2_model_config_by_name(model_name: str) -> dict:
    new_config = DEFAULT_GPT_CONFIG.copy()
    new_config.update(MODEL_CONFIGS[model_name]["config"])
    # Ensure context_length is set to GPT-2 standard
    new_config["context_length"] = 1024

    return new_config
