# 1) implement / return a hugging face transformer model
# 2) write the layer operations more manual
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
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
    ):
        """Tokenize input text"""
        return self.tokenizer(
            input, padding=padding, truncation=truncation, return_tensors=return_tensors
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
