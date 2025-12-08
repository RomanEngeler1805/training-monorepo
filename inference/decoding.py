# greedy decoding -> take the current max logprob; same as n_beams = 1
# beam search -> additional dimension to batch_size; could flatten it and then unflatten to not change all internals
# the decoder is passed to the model, and becomes a class attribute
#
import torch


class GreedyDecoder:
    def __init__(self):
        pass

    def decode(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding of sequence

        inputs:
        - input_ids: torch.Tensor (batch_size, seq_length), token ids of sequence
        - logits: torch.Tensor (batch_size, seq_length, n_vocab), unnormalised probability distribution

        outputs:
        - token_ids: torch.Tensor of shape batch_size, seq_length+1, token ids of sequence plus next token id
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2D (batch_size, vocab_size) or 3D (batch_size, seq_length, vocab_size), "
                f"got {logits.dim()}D tensor with shape {logits.shape}"
            )

        next_token_id = torch.argmax(logits, dim=-1)

        return torch.cat((input_ids, next_token_id.unsqueeze(-1)), dim=-1)
