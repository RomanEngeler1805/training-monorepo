# greedy decoding -> take the current max logprob; same as n_beams = 1
# beam search -> additional dimension to batch_size; could flatten it and then unflatten to not change all internals
# the decoder is passed to the model, and becomes a class attribute
#
import torch


class GreedyDecoder:
    def __init__(self):
        pass

    def reset(self):
        pass

    def decode(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding of sequence

        inputs:
        - input_ids: torch.Tensor (batch_size, seq_length), token ids of sequence
        - logits: torch.Tensor (batch_size, seq_length, n_vocab), unnormalised probability distribution

        outputs:
        - token_ids: torch.Tensor (batch_size, seq_length+1), token ids of sequence plus next token id
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


class BeamDecoder:
    def __init__(self, num_beams: int = 5):
        self.num_beams = num_beams
        self.beam_scores: torch.Tensor | None = None
        self.batch_size: int | None = None

    def reset(self):
        """Reset beam scores for new generation"""
        self.beam_scores = None
        self.batch_size = None

    def decode(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Beam search with num_beams

        inputs:
        - input_ids: torch.Tensor (batch_size, seq_length) on first call,
                     (batch_size * num_beams, seq_length) on subsequent calls
        - logits: torch.Tensor (batch_size, seq_length, n_vocab) on first call,
                  (batch_size * num_beams, seq_length, n_vocab) on subsequent calls

        outputs:
        - token_ids: torch.Tensor (batch_size * num_beams, seq_length+1), token ids of sequence plus next token id
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2D (batch_size, vocab_size) or 3D (batch_size, seq_length, vocab_size), "
                f"got {logits.dim()}D tensor with shape {logits.shape}"
            )

        batch_size_num_beams, vocab_size = logits.shape
        seq_length = input_ids.shape[1]

        # Handle first call: expand (batch_size, seq) to (batch_size * num_beams, seq)
        if self.batch_size is None:
            # First call: input_ids is (batch_size, seq_length), need to expand
            if batch_size_num_beams % self.num_beams != 0:
                # This is the first call - expand the batch
                self.batch_size = batch_size_num_beams
                # Expand input_ids: (batch, seq) -> (batch * beams, seq)
                input_ids = input_ids.repeat_interleave(self.num_beams, dim=0)
                # Expand logits: (batch, vocab) -> (batch * beams, vocab)
                logits = logits.repeat_interleave(self.num_beams, dim=0)
                # Initialize beam scores to 0 (log probability of 1)
                self.beam_scores = torch.zeros(
                    self.batch_size, self.num_beams, device=input_ids.device, dtype=torch.float32
                )
            else:
                # Already expanded (shouldn't happen on first call, but handle it)
                self.batch_size = batch_size_num_beams // self.num_beams
                self.beam_scores = torch.zeros(
                    self.batch_size, self.num_beams, device=input_ids.device, dtype=torch.float32
                )

        # logits: (batch_size, num_beams, vocab_size)
        logits = logits.view(self.batch_size, self.num_beams, vocab_size)
        input_ids = input_ids.view(self.batch_size, self.num_beams, seq_length)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Calculate candidate scores (batch_size, num_beams, vocab_size)
        assert self.beam_scores is not None, "beam_scores should be initialized by this point"
        candidate_scores = self.beam_scores.unsqueeze(-1) + log_probs

        # Select top num_beams; first flatten to (batch_size, num_beams x vocab_size)
        candidate_scores = candidate_scores.view(self.batch_size, -1)
        top_scores, top_indices = torch.topk(candidate_scores, k=self.num_beams, dim=-1)

        # Find which beam this came from and which next token was chosen
        # in order to concatenate the sequence and the next token
        beam_indices = top_indices // vocab_size
        vocab_indices = top_indices % vocab_size

        batch_idx = torch.arange(self.batch_size, device=input_ids.device)
        # Use gather to select: input_ids[batch_idx, beam_indices, :]
        selected_sequences = input_ids[batch_idx.unsqueeze(1), beam_indices]  # (batch, beams, seq)

        next_tokens = vocab_indices.unsqueeze(-1)  # (batch, num_beams, 1)

        # Concatenate & store
        new_input_ids = torch.cat([selected_sequences, next_tokens], dim=-1)
        self.beam_scores = top_scores

        # Flatten back to (batch_size * num_beams, seq_length+1)
        return new_input_ids.view(self.batch_size * self.num_beams, -1)
