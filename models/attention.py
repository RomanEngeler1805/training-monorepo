from math import sqrt

import torch
import torch.nn.functional as F
import torch.nn.init as init


class Attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("Model dimension needs to be divisible by number of heads")

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.w_q = torch.nn.Parameter(torch.empty(d_model, d_model))
        init.xavier_uniform_(self.w_q)
        self.w_k = torch.nn.Parameter(torch.empty(d_model, d_model))
        init.xavier_uniform_(self.w_k)
        self.w_v = torch.nn.Parameter(torch.empty(d_model, d_model))
        init.xavier_uniform_(self.w_v)
        self.w_o = torch.nn.Parameter(torch.empty(d_model, d_model))
        init.xavier_uniform_(self.w_o)

        if device is not None:
            self.to(device)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the input tensor into multiple heads

        Input: (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_length, _ = x.size()
        # Reshape to (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        # Permute to (batch, num_heads, seq_len, d_k) for parallel computation
        # TODO: understand the permutation
        return x.permute(0, 2, 1, 3)

    def concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the input tensor from multiple heads

        Input: (batch, num_heads, seq_len, d_k)
        Output: (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        # Permute back to (batch, seq_len, num_heads, d_k)
        x = x.permute(0, 2, 1, 3)
        # Reshape to (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        inputs:
            x: tensor of shape [batch_size, seq_length, d_model]

        return:
            ?
        """

        # calculate tensors
        # 1) split into individual heads -> calculate attention
        q = self.split_heads(x @ self.w_q)
        k = self.split_heads(x @ self.w_k)
        v = self.split_heads(x @ self.w_v)

        # attention
        # 2) concatenate head outputs
        attn_output = F.softmax(q @ k.transpose(-2, -1) / sqrt(self.d_k), dim=-1) @ v

        # 3) linear layer for information flow
        output = self.concat_heads(attn_output) @ self.w_o

        return output
