import pytest
import torch

from src.models.custom_model import CustomModel


# CustomModel fixtures
@pytest.fixture
def n_layers():
    return 3


@pytest.fixture
def n_vocab():
    return 524288


@pytest.fixture
def d_model():
    return 128


@pytest.fixture
def num_heads():
    return 16


@pytest.fixture
def d_hidden():
    return 256


@pytest.fixture
def custom_model(n_layers, n_vocab, d_model, num_heads, d_hidden):
    return CustomModel(
        n_layers=n_layers,
        n_vocab=n_vocab,
        d_model=d_model,
        num_heads=num_heads,
        d_hidden=d_hidden,
    )


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_length():
    return 5


@pytest.fixture
def input_ids(batch_size, seq_length, n_vocab):
    return torch.randint(low=0, high=n_vocab, size=(batch_size, seq_length))


class TestCustomhModel:
    def test_init(self, custom_model, n_layers):
        assert custom_model is not None
        assert custom_model.embedding is not None
        assert len(custom_model.transformer_blocks) == n_layers
        assert custom_model.decoder is not None

    def test_forward(self, custom_model, input_ids, batch_size, seq_length, n_vocab):
        output = custom_model.forward(input_ids)

        assert output.logits.shape == (batch_size, seq_length, n_vocab)
        assert output.logits.dtype == torch.bfloat16  # Change this to match model dtype
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()
        # Output is logits (raw scores), not probabilities

        # If you want to verify probabilities, apply softmax
        probs = torch.softmax(output.logits, dim=-1)
        assert (probs >= 0).all() and (probs <= 1).all()
        # Check that probabilities sum to ~1 for each position
        assert torch.allclose(
            probs.sum(dim=-1),
            torch.ones(batch_size, seq_length, device=custom_model.device, dtype=torch.bfloat16),
            atol=1e-2,
        )

    def test_train(self, custom_model):
        custom_model.train()
        assert True

    def test_eval(self, custom_model):
        custom_model.eval()
        assert True

    def test_generate(self, custom_model, input_ids, batch_size, seq_length, n_vocab):
        """Test the generate() method produces correct output shape and length"""
        max_length = 10
        initial_seq_length = input_ids.shape[1]

        # Generate tokens
        generated_ids = custom_model.generate(input_ids=input_ids, max_length=max_length)

        # Check output shape
        assert generated_ids.shape == (batch_size, max_length)
        assert generated_ids.shape[1] == max_length

        # Check that input_ids are preserved at the beginning
        assert torch.equal(generated_ids[:, :initial_seq_length].cpu(), input_ids.cpu())

        # Check that generated tokens are valid (within vocab range)
        assert (generated_ids >= 0).all()
        assert (generated_ids < n_vocab).all()

        # Check that model is in eval mode after generation
        assert not custom_model.training

    def test_generate_with_beam_decoder(self, custom_model, input_ids, batch_size, n_vocab):
        """Test generate() method with BeamDecoder as alternative decoder."""
        from src.inference.beam_decoder import BeamDecoder

        max_length = 10
        num_beams = 3
        initial_seq_length = input_ids.shape[1]

        # Create beam decoder
        beam_decoder = BeamDecoder(num_beams=num_beams)

        # Generate tokens with beam search
        generated_ids = custom_model.generate(
            input_ids=input_ids, max_length=max_length, decoder=beam_decoder
        )

        # With beam search, output should be (batch_size * num_beams, max_length)
        assert generated_ids.shape == (batch_size * num_beams, max_length)

        # Check that input_ids are preserved at the beginning for all beams
        for i in range(batch_size):
            for beam in range(num_beams):
                beam_idx = i * num_beams + beam
                assert torch.equal(
                    generated_ids[beam_idx, :initial_seq_length].cpu(), input_ids[i].cpu()
                )

        # Check that generated tokens are valid (within vocab range)
        assert (generated_ids >= 0).all()
        assert (generated_ids < n_vocab).all()

        # Check that model is in eval mode after generation
        assert not custom_model.training
