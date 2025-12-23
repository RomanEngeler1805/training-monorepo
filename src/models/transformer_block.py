import torch

from src.models.attention import Attention
from src.models.feedforward import FeedForward


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
            d_model=d_model, d_hidden=d_hidden, device=device, dropout=dropout, dtype=dtype
        )

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(dropout)

        if device is not None:
            self.to(device)
        self.to(dtype)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass"""
        # Self Attention - PRE-NORM
        x = x + self.dropout(self.multi_head_attention(self.layer_norm1(x), attention_mask))

        # Feed Forward - PRE-NORM
        x = x + self.dropout(self.feed_forward(self.layer_norm2(x)))

        return x
