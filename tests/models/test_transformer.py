import pytest
import torch

from src.models.transformer import Model, ScratchModel, Tokenizer


# ScratchModel fixtures
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
def scratch_model(n_layers, n_vocab, d_model, num_heads, d_hidden):
    return ScratchModel(
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


# Model fixtures (HuggingFace model)
@pytest.fixture
def model_name():
    return "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture
def hf_model(model_name):
    return Model(model_name=model_name)


@pytest.fixture
def attention_mask(batch_size, seq_length):
    return torch.ones(batch_size, seq_length, dtype=torch.long)


# Tokenizer fixtures
@pytest.fixture
def tokenizer(model_name):
    return Tokenizer(tokenizer_name=model_name)


class TestScratchModel:
    def test_init(self, scratch_model, n_layers):
        assert scratch_model is not None
        assert scratch_model.embedding is not None
        assert len(scratch_model.transformer_blocks) == n_layers
        assert scratch_model.decoder is not None

    def test_forward(self, scratch_model, input_ids, batch_size, seq_length, n_vocab):
        output = scratch_model.forward(input_ids)

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
            torch.ones(batch_size, seq_length, device=scratch_model.device, dtype=torch.bfloat16),
            atol=1e-2,
        )

    def test_train(self, scratch_model):
        scratch_model.train()
        assert True

    def test_eval(self, scratch_model):
        scratch_model.eval()
        assert True

    def test_generate(self, scratch_model, input_ids, batch_size, seq_length, n_vocab):
        """Test the generate() method produces correct output shape and length"""
        max_length = 10
        initial_seq_length = input_ids.shape[1]

        # Generate tokens
        generated_ids = scratch_model.generate(input_ids=input_ids, max_length=max_length)

        # Check output shape
        assert generated_ids.shape == (batch_size, max_length)
        assert generated_ids.shape[1] == max_length

        # Check that input_ids are preserved at the beginning
        assert torch.equal(generated_ids[:, :initial_seq_length].cpu(), input_ids.cpu())

        # Check that generated tokens are valid (within vocab range)
        assert (generated_ids >= 0).all()
        assert (generated_ids < n_vocab).all()

        # Check that model is in eval mode after generation
        assert not scratch_model.training

    def test_generate_with_beam_decoder(self, scratch_model, input_ids, batch_size, n_vocab):
        """Test generate() method with BeamDecoder as alternative decoder."""
        from src.inference.decoding import BeamDecoder

        max_length = 10
        num_beams = 3
        initial_seq_length = input_ids.shape[1]

        # Create beam decoder
        beam_decoder = BeamDecoder(num_beams=num_beams)

        # Generate tokens with beam search
        generated_ids = scratch_model.generate(
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
        assert not scratch_model.training


class TestModel:
    def test_init(self, hf_model):
        assert hf_model is not None
        assert hf_model.model is not None

    def test_model_parameters(self, hf_model):
        params = list(hf_model.parameters())
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) or hasattr(p, "requires_grad") for p in params)

    def test_model_forward(self, hf_model, input_ids, attention_mask, batch_size, seq_length):
        output = hf_model.forward(input_ids, attention_mask)
        n_vocab = hf_model.model.config.vocab_size

        assert output.logits.shape == (batch_size, seq_length, n_vocab)
        assert output.logits.dtype == torch.bfloat16
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()

        assert not torch.isnan(output.logits).any()

    def test_model_train(self, hf_model):
        hf_model.train()
        assert hf_model.model.training

    def test_model_eval(self, hf_model):
        hf_model.eval()
        assert not hf_model.model.training


class TestTokenizer:
    def test_tokenizer_init(self, tokenizer):
        assert tokenizer is not None
        assert tokenizer.tokenizer is not None

    def test_tokenizer_tokenize(self, tokenizer):
        input_text = "Hello, world!"
        tokenized = tokenizer.tokenize(input_text)
        output_text = tokenizer.decode(tokenized.input_ids[0])

        assert "input_ids" in tokenized
        assert (
            input_text in output_text or output_text in input_text
        )  # May differ due to tokenization

    def test_tokenizer_batch_tokenize(self, tokenizer):
        input_texts = ["Hello, world!", "Goodbye, world!"]
        tokenized = tokenizer.tokenize(input_texts)
        output_texts = tokenizer.batch_decode(tokenized.input_ids)

        assert len(output_texts) == len(input_texts)
        # Verify that decoded texts contain the original words (exact match may differ)
        for original, decoded in zip(input_texts, output_texts, strict=True):
            assert any(word.lower() in decoded.lower() for word in original.split())
