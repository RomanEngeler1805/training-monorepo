import pytest
import torch

from models.attention import Attention


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def num_heads():
    return 16


@pytest.fixture
def attention(d_model, num_heads):
    return Attention(d_model=d_model, num_heads=num_heads, dtype=torch.float32)


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def seq_length():
    return 5


@pytest.fixture
def input_tensor(batch_size, seq_length, d_model):
    return torch.rand(batch_size, seq_length, d_model, dtype=torch.float32)


class TestAttention:
    def test_init(self, attention, d_model, num_heads):
        assert attention.d_model == d_model
        assert attention.num_heads == num_heads
        assert attention.d_k == d_model // num_heads

        assert attention.w_q.shape == (d_model, d_model)
        assert attention.w_k.shape == (d_model, d_model)
        assert attention.w_v.shape == (d_model, d_model)
        assert attention.w_o.shape == (d_model, d_model)

    def test_init_raises_error_when_d_model_less_than_num_heads(self):
        with pytest.raises(ValueError, match="Model dimension needs to be divisible"):
            Attention(d_model=7, num_heads=16)

    def test_split_heads(self, attention, input_tensor, batch_size, seq_length, num_heads, d_model):
        output = attention.split_heads(input_tensor)

        assert output.shape == (batch_size, num_heads, seq_length, d_model // num_heads)
        assert output.dtype == input_tensor.dtype

    def test_concat_heads(self, attention, batch_size, seq_length, num_heads, d_model):
        x = torch.rand(batch_size, num_heads, seq_length, d_model // num_heads)
        output = attention.concat_heads(x)

        assert output.shape == (batch_size, seq_length, d_model)
        assert output.dtype == x.dtype

    def test_split_and_concat_heads_round_trip(self, attention, input_tensor):
        """Test that split_heads and concat_heads are inverse operations."""
        split = attention.split_heads(input_tensor)
        reconstructed = attention.concat_heads(split)

        assert reconstructed.shape == input_tensor.shape
        torch.testing.assert_close(reconstructed, input_tensor)

    def test_forward(self, attention, input_tensor, batch_size, seq_length, d_model):
        output = attention.forward(input_tensor)

        assert output.shape == (batch_size, seq_length, d_model)
        assert output.dtype == input_tensor.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
