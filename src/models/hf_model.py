import torch
from transformers import AutoModelForCausalLM

from src.inference.beam_decoder import BeamDecoder
from src.inference.greedy_decoder import GreedyDecoder


class HFModel:
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
