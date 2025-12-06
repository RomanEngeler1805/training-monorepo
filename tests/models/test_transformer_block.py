import pytest
import torch

from models.transformer_block import TransformerBlock


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def seq_length():
    return 5


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def num_heads():
    return 16


@pytest.fixture
def d_hidden():
    return 512


@pytest.fixture
def transformer_block(d_model, num_heads, d_hidden):
    return TransformerBlock(d_model=d_model, num_heads=num_heads, d_hidden=d_hidden)


class TestTransformerBlock:
    def test_init(self, transformer_block, d_model):
        # Verify components exist and are correct types
        assert transformer_block.multi_head_attention is not None
        assert transformer_block.feed_forward is not None

        # Verify layer norms are initialized with correct dimensions
        assert transformer_block.layer_norm1 is not None
        assert transformer_block.layer_norm2 is not None
        assert transformer_block.layer_norm1.normalized_shape == (d_model,)
        assert transformer_block.layer_norm2.normalized_shape == (d_model,)
        assert isinstance(transformer_block.layer_norm1, torch.nn.LayerNorm)
        assert isinstance(transformer_block.layer_norm2, torch.nn.LayerNorm)

    def test_forward(self, transformer_block, batch_size, seq_length, d_model):
        x = torch.rand(batch_size, seq_length, d_model)
        output = transformer_block.forward(x)

        assert output.shape == (batch_size, seq_length, d_model)
