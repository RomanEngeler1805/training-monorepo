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
    return TransformerBlock(
        d_model=d_model, num_heads=num_heads, d_hidden=d_hidden, dtype=torch.float32
    )


@pytest.fixture
def input_tensor(batch_size, seq_length, d_model):
    return torch.rand(batch_size, seq_length, d_model, dtype=torch.float32)


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

    def test_forward(self, transformer_block, input_tensor, batch_size, seq_length, d_model):
        output = transformer_block.forward(input_tensor)

        assert output.shape == (batch_size, seq_length, d_model)
        assert output.dtype == input_tensor.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_2d_attention_mask(
        self, transformer_block, input_tensor, batch_size, seq_length, d_model
    ):
        """Test forward pass with 2D attention mask (most common: padding masks)."""
        # Create mask where last 2 positions are masked (padding)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        attention_mask[:, 3:] = 0

        output = transformer_block.forward(input_tensor, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_length, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_3d_attention_mask(
        self, transformer_block, input_tensor, batch_size, seq_length, d_model
    ):
        """Test forward pass with 3D attention mask (causal/autoregressive masks)."""
        # Create causal mask (lower triangular)
        attention_mask = torch.tril(
            torch.ones(batch_size, seq_length, seq_length, dtype=torch.long)
        )

        output = transformer_block.forward(input_tensor, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_length, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_mask_affects_output(
        self, transformer_block, input_tensor, batch_size, seq_length, d_model
    ):
        """Test that masking actually affects the output (not just passes through)."""
        # No mask
        output_no_mask = transformer_block.forward(input_tensor, attention_mask=None)

        # With mask (last position masked)
        mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        mask[:, -1] = 0
        output_with_mask = transformer_block.forward(input_tensor, attention_mask=mask)

        # Outputs should be different, especially at masked positions
        assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-6)
