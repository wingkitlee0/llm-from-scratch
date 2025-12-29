import torch

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.modules import TransformerBlock


def main():
    torch.manual_seed(1234)

    # shape = (batch_size, num_tokens, emb_dim)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    out = block(x)

    print(x.shape)
    print(out.shape)


if __name__ == "__main__":
    main()
