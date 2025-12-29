import argparse

import torch
import torch.nn as nn

from llm.gpt2.modules import ExampleDeepNeuralNetwork


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[1.0]])

    loss = nn.MSELoss()(output, target)

    # Backward pass to compute gradients
    loss.backward()

    print("Layer Gradients without shortcut:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.grad.abs().mean().item()}")


def main(num_layers: int, use_shortcut: bool):
    layer_sizes = [3] * num_layers + [1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=use_shortcut
    )
    print_gradients(model_without_shortcut, sample_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num_layers", type=int, default=6, help="Number of layers"
    )
    parser.add_argument(
        "-s", "--use_shortcut", action="store_true", help="Use shortcut"
    )
    args = parser.parse_args()
    main(
        num_layers=args.num_layers,
        use_shortcut=args.use_shortcut,
    )
