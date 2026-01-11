from typing import Dict

import numpy as np
import tiktoken

from llm.configs.gpt_config import DEFAULT_GPT_CONFIG


def tokenize_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Ray Data map_batches function.
    Tokenizes text and creates sliding windows of (input, target).

    Args:
        batch: Pandas DataFrame or Dict containing 'text' column.

    Returns:
        Dict with 'input_ids' and 'labels'.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    # 'text' column might be missing if we don't select it, but we assume it's there.
    # We must handle cases where 'text' is None or not string if dirty data.
    texts = batch["text"]

    all_input_ids = []
    all_labels = []

    max_length = DEFAULT_GPT_CONFIG["context_length"]  # 1024
    stride = max_length  # Non-overlapping for standard pretraining

    for text in texts:
        if not isinstance(text, str):
            continue

        # Simple tokenization
        try:
            tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        except Exception:
            continue

        # We append <|endoftext|> to separate documents if desired,
        # but here we just process the text as is.
        # Often it is good practice to add EOT token.
        tokens.append(tokenizer.eot_token)

        # Create chunks
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i : i + max_length]
            target_chunk = tokens[i + 1 : i + max_length + 1]

            # We need targets for the last token too.
            # If target_chunk is shorter than max_length, we drop it or pad it.
            # Standard GPT-2 training usually drops the remainder or pads.
            # Here we drop if not full length to keep it simple and consistent.
            if len(input_chunk) == max_length and len(target_chunk) == max_length:
                all_input_ids.append(input_chunk)
                all_labels.append(target_chunk)

    import numpy as np

    # If no results, Ray handles empty batches gracefully usually,
    # but let's ensure we return empty arrays with correct types if empty
    if not all_input_ids:
        return {
            "input_ids": np.array([], dtype=np.int64),
            "labels": np.array([], dtype=np.int64),
        }

    return {
        "input_ids": np.array(all_input_ids, dtype=np.int64),
        "labels": np.array(all_labels, dtype=np.int64),
    }
