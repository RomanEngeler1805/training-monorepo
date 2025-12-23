# 1) implement / return a hugging face transformer model
# 2) write the layer operations more manual
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference.decoding import BeamDecoder, GreedyDecoder
from src.models.embeddings import Embeddings
from src.models.transformer_block import TransformerBlock


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


class Model:
    def __init__(self, model_name: str, dtype: torch.dtype = torch.bfloat16):
        """Initialize HuggingFace model wrapper.

        Args:
            model_name: Name or path of the pretrained model.
            dtype: Data type for model parameters (default: torch.bfloat16).
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        self.dtype = dtype

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

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        decoder: GreedyDecoder | BeamDecoder | None = None,
        num_beams: int = 1,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate text using HuggingFace's generate method.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            max_length: Maximum total length (prompt + completion) - used if max_new_tokens is None
            decoder: Decoder instance (for compatibility, but not used - HuggingFace handles decoding)
            num_beams: Number of beams for beam search (default: 1 for greedy)
            max_new_tokens: Maximum number of new tokens to generate (preferred over max_length)
            temperature: Sampling temperature (higher = more diverse, lower = more deterministic).
                        Only used if do_sample=True. Default 1.0.
            do_sample: If True, use sampling instead of deterministic decoding. Default False.
            attention_mask: Attention mask for input_ids. If None, will be inferred (may cause warnings).

        Returns:
            Generated token IDs of shape (batch_size * num_beams, seq_length)
        """
        self.model.eval()
        with torch.inference_mode():
            # Use num_beams from decoder if provided, otherwise use num_beams parameter
            if decoder is not None and hasattr(decoder, "num_beams"):
                num_beams = decoder.num_beams

            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask,
                max_length=max_length if max_new_tokens is None else None,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,  # Return all beams, not just the best one
                do_sample=do_sample,  # Use deterministic beam search
                temperature=temperature if temperature is not None else None,
                early_stopping=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                pad_token_id=self.model.config.eos_token_id,  # Use EOS as pad token
            )

            # Extract sequences from BeamSearchDecoderOnlyOutput
            generated = outputs.sequences if hasattr(outputs, "sequences") else outputs
            return generated


class Tokenizer:
    def __init__(self, tokenizer_name: str):
        """Initialize tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Set padding side for generation

    def tokenize(
        self,
        input: str | list[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        max_length: int | None = None,
        apply_chat_template: bool = False,
    ):
        """Tokenize input text

        Args:
            input: Input text or list of texts to tokenize
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            max_length: Maximum sequence length
            apply_chat_template: If True, apply chat template (for instruction-following models like Gemma).
                                Converts plain text to chat format automatically.
                                Default False for compatibility with pre-training data.
        """
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        if apply_chat_template:
            # Convert string/list of strings to chat message format and apply template
            if isinstance(input, str):
                # Single prompt: apply template to get formatted string
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": input}],
                    add_generation_prompt=True,  # to add '<start_of_turn>model' at the end
                    tokenize=False,
                )
                # Then tokenize the formatted string
                return self.tokenizer(
                    formatted,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors,
                    max_length=max_length,
                    add_special_tokens=False,  # Template already does
                )
            else:
                # Batch: apply template to each prompt to get formatted strings
                formatted_inputs = []
                for prompt in input:
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,  # to add '<start_of_turn>model' at the end
                        tokenize=False,
                    )
                    formatted_inputs.append(formatted)
                # Then tokenize all formatted strings together (handles batching and padding)
                return self.tokenizer(
                    formatted_inputs,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors,
                    max_length=max_length,
                    add_special_tokens=False,  # Template already does
                )

        # Standard tokenization without chat template (for pre-training compatibility)
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
        """
        Decode a batch of token IDs back to text

        Args:
            token_ids: Token IDs to decode

        Returns:
            List of decoded strings (one per sequence in the batch)
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
