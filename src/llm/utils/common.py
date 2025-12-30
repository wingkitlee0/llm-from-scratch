import tiktoken

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.gpt2.models import GPTModel

DEFAULT_CONFIG_NAME = "124m"

GPT_CONFIGS = {
    DEFAULT_CONFIG_NAME: GPT_CONFIG_124M,
}


def get_gpt_model(config_name: str = DEFAULT_CONFIG_NAME) -> GPTModel:
    return GPTModel(GPT_CONFIGS[config_name])


def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")
