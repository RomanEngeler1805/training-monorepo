import torch

from models.attention import Attention


class TestAttention:
    def test_init(self):
        d_model = 64
        attention = Attention(d_model=d_model, d_q=4, d_k=4, d_v=8)

        assert attention.w_q is not None
        assert attention.w_k is not None
        assert attention.w_v is not None

    def test_forward(self):
        batch_size = 2
        d_model = 64
        x = torch.rand(batch_size, d_model)

        attention = Attention(d_model=d_model, d_q=4, d_k=4, d_v=8)
        y = attention.forward(x)

        assert y is not None
