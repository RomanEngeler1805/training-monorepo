# 1) implement / return a hugging face transformer model
# 2) write the layer operations more manual
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.decoding import GreedyDecoder
from models.embeddings import Embeddings
from models.transformer_block import TransformerBlock


@dataclass
class ModelOutput:
    logits: torch.Tensor


class ScratchModel(torch.nn.Module):
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
        self, input_ids: torch.Tensor, max_length: int = 50, decoder: GreedyDecoder | None = None
    ) -> torch.Tensor:
        """Generate text using the provided decoder strategy"""
        if decoder is None:
            decoder = GreedyDecoder()

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


class Model:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        self.device = next(self.model.parameters()).device

    def parameters(self):
        """Return model parameters for optimizer"""
        return self.model.parameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Forward pass through the model

        returns:
        - logits: torch.Tensor, shape (batch_size, sequence_length, vocab_size)
        - loss: torch.Tensor, shape (1,)
        - hidden_states: torch.Tensor, shape (batch_size, sequence_length, hidden_size)
        - attentions: torch.Tensor, shape (batch_size, num_heads, sequence_length, sequence_length)
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if attention_mask is None:
            return self.model(input_ids)
        else:
            return self.model(input_ids, attention_mask=attention_mask)

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()


class Tokenizer:
    def __init__(self, tokenizer_name: str):
        """Initialize tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(
        self,
        input: str | list[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        max_length: int | None = None,
    ):
        """Tokenize input text"""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            input,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            max_length=max_length,
        )

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode a batch of token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
