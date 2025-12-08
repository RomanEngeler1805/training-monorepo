import torch
from torch.nn import CrossEntropyLoss


class CrossEntropy:
    def __init__(self):
        self.loss = CrossEntropyLoss()

    def calculate_loss(
        self, preds: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        """Calculate the Cross Entropy loss

        Args:
            preds: Logits tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Target class indices tensor of shape (batch_size, sequence_length)
                   Must be integer type (dtype=torch.long)
             attention_mask: Optional mask tensor of shape (batch_size, sequence_length)
                          where 1 = valid token, 0 = padding token

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
        indices = torch.arange(batch_size * seq_len, device=preds.device)
        token_losses = -log_probs[indices, labels_flattened]

        # Mask out padding tokens if mask is provided
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(batch_size * seq_len).to(preds.device)
            token_losses = token_losses * mask_flat
            mask_sum = mask_flat.sum()
            if mask_sum > 0:
                loss = token_losses.sum() / mask_sum
            else:
                # Edge case: all tokens masked
                loss = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        else:
            loss = token_losses.mean()

        del preds_flattened, labels_flattened, max_logits, log_probs

        # efficient loss calculation via indexing
        return loss
