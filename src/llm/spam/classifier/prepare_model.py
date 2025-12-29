import torch

from llm.gpt2.models import GPTModel


def prepare_gpt2_for_classification(
    model: GPTModel, model_config: dict, device: torch.device
) -> torch.nn.Module:
    """Prepare a GPT-2 model for classification."""
    assert isinstance(model, GPTModel), "Model must be a GPT-2 model"
    # Add a classification layer
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=model_config["emb_dim"],
        out_features=num_classes,
        device=device,
    )

    # Ensure out_head parameters are trainable
    for param in model.out_head.parameters():
        param.requires_grad = True

    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model
