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

        # NLL = - sum_t(log p_theta(x_t | x_<t)) -> so can parallalize this

        # Reshape tensors to (batch_size x seq_length, vocab_size)
        preds_flat = preds.reshape(batch_size * seq_len, vocab_size)

        # Get probability over vocabulary -> e^x_j / sum_i(e^x_i)
        # take log for stabilits -> x_j - log(sum(e^x_i))
        max_logits = torch.max(preds_flat, dim=1, keepdim=True)[0]
        log_probs = (
            preds_flat
            - max_logits
            - torch.log(torch.sum(torch.exp(preds_flat - max_logits), dim=1, keepdim=True))
        )

        # Extract the probabilities for x_t (teacher forcing)
        indices = torch.arange(batch_size * seq_len, device=preds.device)
        log_probs = log_probs[
            indices, labels.reshape(-1)
        ]  # (batch_size x seq_len, vocab_size) -> (batch_size x seq_len)
        token_loss = -log_probs

        # Token loss with attention mask
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1, 1).to(preds.device)
            mask_sum = mask_flat.sum()
            if mask_sum > 0:
                loss = token_loss.sum() / mask_sum
            else:
                loss = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        else:
            loss = token_loss.mean()

        # efficient loss calculation via indexing
        return loss
