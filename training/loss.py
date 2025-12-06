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

        # variables
        batch_size, seq_len, vocab_size = preds.shape

        # flatten
        preds_flattened = preds.reshape(batch_size * seq_len, vocab_size)
        labels_flattened = labels.reshape(batch_size * seq_len)

        # prepare logprobs. subtract max for numerical stability
        max_logits = torch.max(preds_flattened, dim=1, keepdim=True)[0]
        log_probs = (
            preds_flattened
            - max_logits
            - torch.log(torch.sum(torch.exp(preds_flattened - max_logits), dim=1, keepdim=True))
        )

        # naive implementation that explodes tensors
        # import torch.nn.functional as F
        # labels_onehot = F.one_hot(labels_flattened, num_classes=vocab_size)
        # loss = - (log_probs * labels_onehot).mean()
        loss = -log_probs[torch.arange(batch_size * seq_len), labels_flattened].mean()
        del preds_flattened, labels_flattened, max_logits, log_probs

        # efficient loss calculation via indexing
        return loss
