import pytest
import torch

from models.embeddings import Embeddings


@pytest.fixture
def n_vocab():
    return 1024


@pytest.fixture
def d_embedding():
    return 128


@pytest.fixture
def embedding(n_vocab, d_embedding):
    return Embeddings(n_vocab=n_vocab, d_embedding=d_embedding, dtype=torch.float32)


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def seq_length():
    return 5


@pytest.fixture
def input_ids(batch_size, seq_length, n_vocab):
    return torch.randint(low=0, high=n_vocab, size=(batch_size, seq_length))


class TestEmbeddings:
    def test_init(self, embedding, n_vocab, d_embedding):
        assert embedding.d_embedding == d_embedding
        assert embedding.w.shape == (n_vocab, d_embedding)
        assert isinstance(embedding.w, torch.nn.Parameter)

    def test_forward(self, embedding, input_ids, batch_size, seq_length, d_embedding):
        output = embedding.forward(input_ids)

        assert output.shape == (batch_size, seq_length, d_embedding)
        assert output.dtype == embedding.w.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_embedding_correctness(self, embedding, n_vocab, d_embedding):
        """Test that forward correctly retrieves embeddings for given token IDs."""
        # Create input with known token IDs
        input_ids = torch.tensor([[0, 1, n_vocab - 1]])
        output = embedding.forward(input_ids)

        # Verify that embeddings match the weight matrix rows (accounting for positional encoding)
        # output = token_embedding + positional_encoding, so we subtract pos_encoding to get token_embedding
        token_embedding_0 = (
            output[0, 0] / (embedding.d_embedding**0.5) - embedding.pos_encoding[0, 0, :]
        )
        token_embedding_1 = (
            output[0, 1] / (embedding.d_embedding**0.5) - embedding.pos_encoding[0, 1, :]
        )
        token_embedding_2 = (
            output[0, 2] / (embedding.d_embedding**0.5) - embedding.pos_encoding[0, 2, :]
        )

        torch.testing.assert_close(token_embedding_0, embedding.w[0])
        torch.testing.assert_close(token_embedding_1, embedding.w[1])
        torch.testing.assert_close(token_embedding_2, embedding.w[n_vocab - 1])
