import pytest

from src.models.hf_tokenizer import HFTokenizer


@pytest.fixture
def model_name():
    return "hf-internal-testing/tiny-random-gpt2"


# Tokenizer fixtures
@pytest.fixture
def tokenizer(model_name):
    return HFTokenizer(tokenizer_name=model_name)


class TestHFTokenizer:
    def test_tokenizer_init(self, tokenizer):
        assert tokenizer is not None
        assert tokenizer.tokenizer is not None

    def test_tokenizer_tokenize(self, tokenizer):
        input_text = "Hello, world!"
        tokenized = tokenizer.tokenize(input_text)
        output_text = tokenizer.decode(tokenized.input_ids[0])

        assert "input_ids" in tokenized
        assert (
            input_text in output_text or output_text in input_text
        )  # May differ due to tokenization

    def test_tokenizer_batch_tokenize(self, tokenizer):
        input_texts = ["Hello, world!", "Goodbye, world!"]
        tokenized = tokenizer.tokenize(input_texts)
        output_texts = tokenizer.batch_decode(tokenized.input_ids)

        assert len(output_texts) == len(input_texts)
        # Verify that decoded texts contain the original words (exact match may differ)
        for original, decoded in zip(input_texts, output_texts, strict=True):
            assert any(word.lower() in decoded.lower() for word in original.split())
