import torch
from torch.nn import CrossEntropyLoss


class CrossEntropy:
    def __init__(self):
        self.loss = CrossEntropyLoss()

    def calculate_loss(self, preds: torch.Tensor, labels: torch.Tensor):
        """Calculate the Cross Entropy loss

        Args:
            preds: Logits tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Target class indices tensor of shape (batch_size, sequence_length)
                   Must be integer type (dtype=torch.long)

        Returns:
            Scalar tensor containing the cross entropy loss
        """
        if preds.dim() != 3:
            raise ValueError(
                f"preds must be 3D tensor (batch_size, sequence_length, vocab_size), "
                f"got {preds.dim()}D tensor with shape {preds.shape}"
            )

        # calculate the loss
        batch_size, seq_len, vocab_size = preds.shape
        return self.loss(
            preds.reshape(batch_size * seq_len, vocab_size), labels.reshape(batch_size * seq_len)
        )
