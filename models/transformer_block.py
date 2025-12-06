import torch

from models.attention import Attention
from models.feedforward import FeedForward


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        dropout: float = 0.1,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.multi_head_attention = Attention(
            d_model=d_model, num_heads=num_heads, device=device, dtype=dtype
        )
        self.feed_forward = FeedForward(
            d_model=d_model, d_hidden=d_hidden, device=device, dtype=dtype
        )

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(dropout)

        if device is not None:
            self.to(device)
        self.to(dtype)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass"""
        # Self Attention
        attn_output = self.multi_head_attention(x, attention_mask)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm1(attn_output + x)

        # Feed Forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm2(ff_output + x)

        return x
