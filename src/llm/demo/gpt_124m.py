import torch

from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.demo.dummy_batch import get_dummy_batch
from llm.gpt2.models import GPTModel


def get_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(1234)

    model = GPTModel(GPT_CONFIG_124M)

    batch = get_dummy_batch()
    print(batch[:2])

    out = model(batch[:2])

    print(out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    # Below are the numbers IF we apply weight-tying. It is
    # not currently implemented in the code.
    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )
    print(
        f"Number of trainable parameters "
        f"considering weight tying: {total_params_gpt2:,}"
    )

    num_params_transformer_blocks = sum(
        [get_num_params(block) for block in model.transformer_blocks]
    )
    print(
        f"Number of trainable parameters in transformer blocks: {num_params_transformer_blocks}"
    )


if __name__ == "__main__":
    main()
