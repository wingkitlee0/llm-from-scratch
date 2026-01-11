import torch

from llm.configs.gpt_config import DEFAULT_GPT_CONFIG
from llm.gpt2.modules import FeedForward


def main():
    ffn = FeedForward(DEFAULT_GPT_CONFIG)
    x = torch.randn(2, 3, DEFAULT_GPT_CONFIG["emb_dim"])
    out = ffn(x)
    print(out.shape)


if __name__ == "__main__":
    main()
