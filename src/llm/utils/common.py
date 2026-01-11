import tiktoken

from llm.configs.gpt_config import DEFAULT_GPT_CONFIG
from llm.gpt2.models import GPTModel

DEFAULT_CONFIG_NAME = "124m"

GPT_CONFIGS = {
    DEFAULT_CONFIG_NAME: DEFAULT_GPT_CONFIG,
}


def get_gpt_model(config_name: str = DEFAULT_CONFIG_NAME) -> GPTModel:
    return GPTModel(GPT_CONFIGS[config_name])


def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")
