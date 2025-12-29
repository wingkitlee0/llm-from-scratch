import argparse
import os
from typing import Optional

import torch

from llm.common import get_tokenizer
from llm.gpt2.pretrained import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG_KEYS,
    get_gpt2_model_with_weights,
)
from llm.spam.padding import Padding


def load_finetuned_model(model_path: str, model_name: str, device: torch.device):
    """Load a finetuned model from a checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]

    # Reconstruct base model with pretrained weights
    model, _ = get_gpt2_model_with_weights(model_name=model_name)

    # Replace output head with classification head (matching training setup)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=model_config["emb_dim"],
        out_features=num_classes,
        device=device,
    )

    # Load finetuned weights - check for mismatches
    state_dict = checkpoint["model_state_dict"]
    model_state_dict = model.state_dict()

    # Check for key mismatches
    missing_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    unexpected_keys = set(model_state_dict.keys()) - set(state_dict.keys())

    if missing_keys:
        print(f"Warning: Missing keys in model: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in model: {unexpected_keys}")

    # Load with strict=False to handle any minor mismatches
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Warning: Missing keys when loading: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Warning: Unexpected keys when loading: {load_result.unexpected_keys}")

    # Verify out_head weights were loaded
    if "out_head.weight" in state_dict:
        print(
            f"out_head.weight shape in checkpoint: {state_dict['out_head.weight'].shape}"
        )
        print(f"out_head.weight shape in model: {model.out_head.weight.shape}")
        if torch.allclose(model.out_head.weight, state_dict["out_head.weight"]):
            print("✓ out_head weights loaded correctly")
        else:
            print("✗ WARNING: out_head weights may not have loaded correctly!")

    model.to(device)
    model.eval()

    return model, model_config


def classify_text(
    model,
    text: str,
    tokenizer,
    max_length: int,
    device: torch.device,
    padding: Optional[Padding] = None,
):
    """Classify a text as spam or ham."""

    padding = padding or Padding()

    # Tokenize input - use same method as dataset
    encoded = tokenizer.encode(text)

    # Truncate if too long - match SpamDataset logic: use first max_length tokens
    encoded = padding.truncate(encoded, max_length)

    # Pad to max_length to match training format (same as SpamDataset)
    [encoded] = padding.pad_sequences([encoded], max_length)

    token_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    # Get model prediction - mask out padding tokens and use last non-padding token (matches training)
    with torch.no_grad():
        all_logits = model(token_ids)
        logits = padding.extract_logits_at_last_non_padding(all_logits, token_ids)
        probs = torch.softmax(logits, dim=-1)

    # Get predicted class and confidence
    probs_1d = probs.squeeze(0)
    predicted_class_idx = torch.argmax(probs_1d, dim=-1).item()
    predicted_class = int(predicted_class_idx)
    confidence = float(probs_1d[predicted_class].item())

    return predicted_class, confidence, probs_1d


def main(model_path: str, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    padding = Padding()

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model, model_config = load_finetuned_model(model_path, model_name, device)

    # Get max_length from checkpoint (used during training for padding)
    # If not in checkpoint, fall back to context_length (for backward compatibility)
    max_length = checkpoint.get("max_length")
    if max_length is None:
        max_length = model_config.get("context_length", 1024)
        print(
            f"Warning: max_length not found in checkpoint, using context_length: {max_length}"
        )
    else:
        print(f"Using max_length: {max_length} (from training)")
    print("Model loaded successfully!")

    # Test with a known ham example
    print("\nTesting with known examples...")
    test_ham = "Just forced myself to eat a slice. I'm really not hungry tho."
    test_spam = "WELL DONE! Your 4* Costa Del Sol Holiday or £5000 await collection."

    ham_class, ham_confidence, ham_probs = classify_text(
        model, test_ham, tokenizer, max_length, device, padding
    )
    spam_class, spam_conf, spam_probs = classify_text(
        model, test_spam, tokenizer, max_length, device, padding
    )

    print(f"Test HAM text: {test_ham}")
    print(f"  Prediction: {'SPAM' if ham_class == 1 else 'HAM'} (class {ham_class})")
    print(
        f"  Probabilities: HAM {ham_probs[0].item() * 100:.2f}% | SPAM {ham_probs[1].item() * 100:.2f}%"
    )
    print(f"\nTest SPAM text: {test_spam}")
    print(f"  Prediction: {'SPAM' if spam_class == 1 else 'HAM'} (class {spam_class})")
    print(
        f"  Probabilities: HAM {spam_probs[0].item() * 100:.2f}% | SPAM {spam_probs[1].item() * 100:.2f}%"
    )

    # Interactive classification loop
    print("\n" + "=" * 60)
    print("Spam Classifier - Enter text to classify (type 'quit' to exit)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Enter text: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter some text.\n")
                continue

            # Classify
            predicted_class, confidence, probs = classify_text(
                model, user_input, tokenizer, max_length, device, padding
            )

            # Display results
            label = "SPAM" if predicted_class == 1 else "HAM"
            spam_prob = probs[1].item() * 100
            ham_prob = probs[0].item() * 100

            print(f"\nPrediction: {label}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print(f"Probabilities: HAM {ham_prob:.2f}% | SPAM {spam_prob:.2f}%")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a finetuned spam classifier and classify user input"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the saved finetuned model checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=MODEL_CONFIG_KEYS,
        help=f"Base model name. Available models: {', '.join(MODEL_CONFIG_KEYS)}",
    )
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"models/spam_classifier/{args.model}_finetuned.pt"

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train a model first using finetune.py")
        exit(1)

    main(model_path=args.model_path, model_name=args.model)
