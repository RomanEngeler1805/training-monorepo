import torch


class Embeddings(torch.nn.Module):
    def __init__(
        self,
        n_vocab: int,
        d_embedding: int,
        max_seq_len: int = 5000,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # Token embeddings
        self.d_embedding = d_embedding
        self.w = torch.nn.Parameter(torch.empty(n_vocab, d_embedding, device=device, dtype=dtype))
        torch.nn.init.normal_(self.w, mean=0.0, std=0.02)

        # Positional encoding
        self.pos_encoding: torch.Tensor
        self.register_buffer(
            "pos_encoding",
            self._create_positional_encoding(
                max_len=max_seq_len, d_embedding=d_embedding, dtype=dtype
            ),
        )

        # Device handling
        if device is not None:
            self.to(device)
        self.to(dtype)

    def _create_positional_encoding(self, max_len: int, d_embedding: int, dtype: torch.dtype):
        # Allocate space
        pe = torch.zeros(max_len, d_embedding, dtype=dtype)

        # Calculate exponent
        pos = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.pow(10000.0, -torch.arange(0, d_embedding, 2, dtype=dtype) / d_embedding)

        # Calculate positional embeddings
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)

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

        embeddings = self.w[input_ids, :].view(batch_size, seq_length, self.d_embedding)
        embeddings += self.pos_encoding[:, :seq_length, :]

        # Scale embeddings by sqrt(d_model) - common in modern transformers
        embeddings = embeddings * (self.d_embedding**0.5)

        return embeddings
