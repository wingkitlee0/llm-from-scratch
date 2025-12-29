from importlib import resources

import yaml

DEFAULT_MODEL_NAME = "gpt2-small"

# Load model configs from YAML file
with resources.files("llm.configs").joinpath("model_configs.yaml").open("r") as f:
    MODEL_CONFIGS = yaml.safe_load(f)


MODEL_CONFIG_KEYS = list(MODEL_CONFIGS.keys())
