import torch

from models.attention import Attention
from models.feedforward import FeedForward


class TransformerBlock:
    def __init__(self, d_model: int, num_heads: int, d_hidden: int, dropout: float = 0.1):
        self.multi_head_attention = Attention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model=d_model, d_hidden=d_hidden)

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Self Attention
        attn_output = self.multi_head_attention.forward(x)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm1(attn_output + x)

        # Feed Forward
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm2(ff_output + x)

        return x
