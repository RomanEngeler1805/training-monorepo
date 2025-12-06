import torch


class Embeddings(torch.nn.Module):
    def __init__(
        self, n_vocab: int, d_embedding: int, device=None, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.d_embedding = d_embedding
        self.w = torch.nn.Parameter(torch.empty(n_vocab, d_embedding, device=device, dtype=dtype))
        torch.nn.init.xavier_normal_(self.w)

        if device is not None:
            self.to(device)
        self.to(dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed the input tokens into the embedding space

        inputs:
        - input_ids: token_ids of shape [batch_size, seq_length]  # Fixed

        outputs:
        - embeddings of shape [batch_size, seq_length, d_embedding]
        """
        batch_size, seq_length = input_ids.shape
        input_ids = input_ids.reshape(-1)

        embeddings = self.w[input_ids, :]

        return embeddings.view(batch_size, seq_length, self.d_embedding)
