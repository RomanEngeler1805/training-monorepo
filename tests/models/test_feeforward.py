import torch

from models.feedforward import FeedForward


class TestFeeForward:
    def test_init(self):
        d_model = 16
        d_hidden = 32
        layer = FeedForward(d_model=d_model, d_hidden=d_hidden)

        assert layer.w1.shape == (d_model, d_hidden)
        assert layer.b1.shape == (d_hidden,)

        assert layer.w2.shape == (d_hidden, d_model)
        assert layer.b2.shape == (d_model,)

    def test_forward(self):
        batch_size = 4
        seq_length = 8
        d_model = 16
        d_hidden = 32

        layer = FeedForward(d_model=d_model, d_hidden=d_hidden, dtype=torch.float32)
        x = torch.rand(batch_size, seq_length, d_model, dtype=torch.float32)

        output = layer.forward(x)

        assert output.shape == (batch_size, seq_length, d_model)
