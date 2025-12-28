import torch

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.modules import FeedForward


def main():
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.randn(2, 3, GPT_CONFIG_124M["emb_dim"])
    out = ffn(x)
    print(out.shape)


if __name__ == "__main__":
    main()
