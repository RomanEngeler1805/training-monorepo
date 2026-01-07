# 1) implement / return a hugging face transformer model
# 2) write the layer operations more manual
from dataclasses import dataclass

import torch

from src.inference.beam_decoder import BeamDecoder
from src.inference.greedy_decoder import GreedyDecoder
from src.models.embeddings import Embeddings
from src.models.transformer_block import TransformerBlock


@dataclass
class ModelOutput:
    logits: torch.Tensor


class CustomModel(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_vocab: int,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dtype = dtype

        # Device detection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Embedding
        self.embedding = Embeddings(
            n_vocab=n_vocab, d_embedding=d_model, device=self.device, dtype=self.dtype
        )

        # Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_hidden=d_hidden,
                    dropout=dropout,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_layer_norm = torch.nn.LayerNorm(d_model)

        # Output projection (decoder)
        self.decoder = torch.nn.Linear(d_model, n_vocab, bias=False)
        torch.nn.init.xavier_normal_(self.decoder.weight)

        # Move to device and convert to dtype
        self.to(self.device)
        self.to(self.dtype)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> ModelOutput:
        """Forward pass through the model

        inputs:
        - TODO

        returns:
        - logits: torch.Tensor, shape (batch_size, sequence_length, vocab_size)
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        x = self.embedding(input_ids)
        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        x = self.final_layer_norm(x)

        logits = self.decoder(x)
        return ModelOutput(logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        decoder: GreedyDecoder | BeamDecoder | None = None,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        attention_mask: torch.Tensor | None = None,  # TODO: incorporate this properly
    ) -> torch.Tensor:
        """
        Generate text using the provided decoder strategy

        inputs:

        outputs:
        - token_ids: torch.Tensor (batch_size (* num_beams), seq_length)
        """
        if decoder is None:
            decoder = GreedyDecoder()
        decoder.reset()

        self.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            current_ids = input_ids.clone()

            for _ in range(max_length - input_ids.shape[1]):
                output = self.forward(current_ids)

                # Decode next token
                current_ids = decoder.decode(current_ids, output.logits)
        return current_ids

    def train(self, mode: bool = True):
        """Set model to training mode"""
        super().train(mode)
        return self

    def eval(self):
        """Set model to evaluation mode"""
        super().eval()
        return self
