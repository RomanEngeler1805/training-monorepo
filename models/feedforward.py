import torch
import torch.nn as nn
import torch.nn.init as init


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # First linear transformation
        self.w1 = torch.nn.Parameter(torch.empty(d_model, d_hidden, dtype=dtype))
        init.xavier_uniform_(self.w1)
        self.b1 = torch.nn.Parameter(torch.zeros(d_hidden, dtype=dtype))

        # Second linear transformation
        self.w2 = torch.nn.Parameter(torch.empty(d_hidden, d_model, dtype=dtype))
        init.xavier_uniform_(self.w2)
        self.b2 = torch.nn.Parameter(torch.zeros(d_model, dtype=dtype))

        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.activation: nn.Module
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if device is not None:
            self.to(device)
        self.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        inputs:
            x: tensor of shape [batch_size, seq_length, d_model]
        """
        x = self.activation(x @ self.w1 + self.b1)
        x = self.dropout(x)
        x = x @ self.w2 + self.b2
        x = self.dropout(x)
        return x
