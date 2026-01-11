import os

import torch

from llm.spam.classifier.train import TrainResults

DEFAULT_MODEL_DIR = "models/spam_classifier"


def save_classifier_model(
    model: torch.nn.Module,
    model_config: dict,
    max_length: int,
    train_results: TrainResults,
    model_name: str,
    model_dir: str = DEFAULT_MODEL_DIR,
):
    """Save the classifier model."""

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_finetuned.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "max_length": max_length,  # Save max_length for inference
            "train_losses": train_results.train_losses,
            "val_losses": train_results.val_losses,
            "train_accs": train_results.train_accs,
            "val_accs": train_results.val_accs,
            "examples_seen": train_results.examples_seen,
        },
        model_path,
    )
    print(f"Classifier model saved to {model_path}")
