import torch

from llm.configs.gpt_config import DEFAULT_GPT_CONFIG
from llm.gpt2.modules import TransformerBlock


def main():
    torch.manual_seed(1234)

    # shape = (batch_size, num_tokens, emb_dim)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(DEFAULT_GPT_CONFIG)
    out = block(x)

    print(x.shape)
    print(out.shape)


if __name__ == "__main__":
    main()
