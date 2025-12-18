import torch

from src.inference.decoding import BeamDecoder, GreedyDecoder


class TestGreedyDecoder:
    def test_decode(self):
        input_ids = torch.randint(low=0, high=256, size=(8, 5))
        logits = torch.rand(8, 256)

        decoder = GreedyDecoder()
        token_ids = decoder.decode(input_ids=input_ids, logits=logits)

        assert token_ids.shape == (8, 6)


class TestBeamDecoder:
    def test_decode_first_call(self):
        """Test beam decoder on first call (expands batch dimension)."""
        batch_size = 2
        seq_length = 5
        vocab_size = 100
        num_beams = 3

        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length))
        logits = torch.rand(batch_size, vocab_size)

        decoder = BeamDecoder(num_beams=num_beams)
        output_ids = decoder.decode(input_ids=input_ids, logits=logits)

        # Should expand: (batch_size, seq) -> (batch_size * num_beams, seq+1)
        assert output_ids.shape == (batch_size * num_beams, seq_length + 1)
        assert decoder.batch_size == batch_size
        assert decoder.beam_scores is not None
        assert decoder.beam_scores.shape == (batch_size, num_beams)

    def test_decode_subsequent_calls(self):
        """Test beam decoder on subsequent calls (maintains beam dimension)."""
        batch_size = 2
        seq_length = 5
        vocab_size = 100
        num_beams = 3

        decoder = BeamDecoder(num_beams=num_beams)

        # First call
        input_ids_1 = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length))
        logits_1 = torch.rand(batch_size, vocab_size)
        output_ids_1 = decoder.decode(input_ids=input_ids_1, logits=logits_1)

        # Second call (should maintain beam dimension)
        input_ids_2 = output_ids_1  # Use output from first call
        logits_2 = torch.rand(batch_size * num_beams, vocab_size)
        output_ids_2 = decoder.decode(input_ids=input_ids_2, logits=logits_2)

        assert output_ids_2.shape == (batch_size * num_beams, seq_length + 2)
        assert decoder.batch_size == batch_size

    def test_reset(self):
        """Test that reset clears internal state."""
        batch_size = 2
        vocab_size = 100
        num_beams = 3

        decoder = BeamDecoder(num_beams=num_beams)
        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, 5))
        logits = torch.rand(batch_size, vocab_size)

        # First call sets internal state
        decoder.decode(input_ids=input_ids, logits=logits)
        assert decoder.batch_size is not None
        assert decoder.beam_scores is not None

        # Reset should clear state
        decoder.reset()
        assert decoder.batch_size is None
        assert decoder.beam_scores is None
