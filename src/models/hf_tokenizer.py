import torch
from transformers import AutoTokenizer


class HFTokenizer:
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
