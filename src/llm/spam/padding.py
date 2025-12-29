import torch


class Padding:
    def __init__(self, pad_token_id: int = 50256):
        """
        Args:
            pad_token_id: int, the token ID used for padding (default: 50256)
        """
        self.pad_token_id = pad_token_id

    def get_last_non_padding_indices(self, input_batch):
        """
        Find the last non-padding token index for each sequence in a batch by counting the number of padding tokens.
        """
        mask = input_batch != self.pad_token_id
        last_non_pad_indices = (mask.sum(dim=1) - 1).clamp(min=0)
        return last_non_pad_indices

    def extract_logits_at_last_non_padding(self, all_logits, input_batch):
        """
        Extract logits at the last non-padding token position for each sequence in a batch.

        Args:
            all_logits: torch.Tensor of shape (batch_size, seq_len, num_classes) containing
                        logits for all positions
            input_batch: torch.Tensor of shape (batch_size, seq_len) containing token IDs

        Returns:
            torch.Tensor of shape (batch_size, num_classes) containing logits at the last
            non-padding token position for each sequence.
        """
        last_non_pad_indices = self.get_last_non_padding_indices(input_batch)
        batch_indices = torch.arange(input_batch.size(0), device=input_batch.device)
        logits = all_logits[batch_indices, last_non_pad_indices, :]
        return logits

    def pad_sequences(
        self, sequences: list[list[int]], max_length: int
    ) -> list[list[int]]:
        return [
            sequence + [self.pad_token_id] * (max_length - len(sequence))
            for sequence in sequences
        ]

    def truncate(self, sequence: list[int], max_length: int) -> list[int]:
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence
