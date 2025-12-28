import torch
from llm.modules import FeedForward
from llm.configs.gpt_config import GPTConfig

def main():

    ffn = FeedForward(GPTConfig)
    x = torch.randn(2, 3, GPTConfig["emb_dim"])
    out = ffn(x)
    print(out.shape)

if __name__ == "__main__":
    main()