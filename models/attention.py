from math import sqrt

import torch
import torch.nn.functional as F
import torch.nn.init as init


class Attention:
    def __init__(self, d_model: int, d_q: int, d_k: int, d_v: int):
        self.d_k = d_k
        self.w_q = torch.nn.Parameter(torch.empty(d_model, d_q))
        init.xavier_uniform_(self.w_q)
        self.w_k = torch.nn.Parameter(torch.empty(d_model, d_k))
        init.xavier_uniform_(self.w_k)
        self.w_v = torch.nn.Parameter(torch.empty(d_model, d_v))
        init.xavier_uniform_(self.w_v)

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        inputs:
            x: tensor of shape [batch_size, d_model]

        return:
            ?
        """

        # calculate tensors
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        # attention
        output = F.softmax(q @ k.T / sqrt(self.d_k), dim=1) @ v

        return output
