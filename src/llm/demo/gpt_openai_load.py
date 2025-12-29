import torch

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.demo.gpt_download import download_and_load_gpt2
from llm.demo.gpt_load import load_weights_into_gpt
from llm.demo.text_utils import generate, text_to_token_ids, token_ids_to_text
from llm.gpt_models import GPTModel
from llm.top_level import get_tokenizer

DEFAULT_MODEL_NAME = "gpt2-small (124M)"

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def get_gpt_model_openai(model_name: str = DEFAULT_MODEL_NAME) -> tuple[GPTModel, dict]:
    new_config = GPT_CONFIG_124M.copy()
    new_config.update(MODEL_CONFIGS[model_name])
    new_config.update({"context_length": 1024})
    new_config.update({"qkv_bias": True})

    return GPTModel(new_config), new_config


def main():
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    model, model_config = get_gpt_model_openai()
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
    main()
