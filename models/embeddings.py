import torch


class Embeddings:
    def __init__(self, n_vocab: int, d_embedding: int):
        self.d_embedding = d_embedding
        self.w = torch.nn.Parameter(torch.empty(n_vocab, d_embedding))
        torch.nn.init.xavier_normal_(self.w)

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
