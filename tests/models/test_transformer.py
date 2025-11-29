import torch

from models.transformer import Model, Tokenizer


class TestModel:
    def test_model_init(self):
        model = Model(model_name="hf-internal-testing/tiny-random-gpt2")
        assert model is not None

    def test_model_parameters(self):
        model = Model(model_name="hf-internal-testing/tiny-random-gpt2")
        assert model.parameters() is not None

    def test_model_forward(self):
        model = Model(model_name="hf-internal-testing/tiny-random-gpt2")
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        output = model.forward(input_ids, attention_mask)
        vocab_size = model.model.config.vocab_size
        assert output.logits.shape == (1, 3, vocab_size)

    def test_model_train(self):
        model = Model(model_name="hf-internal-testing/tiny-random-gpt2")
        model.train()
        assert model.model.training

    def test_model_eval(self):
        model = Model(model_name="hf-internal-testing/tiny-random-gpt2")
        model.eval()
        assert not model.model.training


class TestTokenizer:
    def test_tokenizer_init(self):
        tokenizer = Tokenizer(tokenizer_name="hf-internal-testing/tiny-random-gpt2")
        assert tokenizer is not None

    def test_tokenizer_tokenize(self):
        tokenizer = Tokenizer(tokenizer_name="hf-internal-testing/tiny-random-gpt2")
        input_text = "Hello, world!"
        tokenized = tokenizer.tokenize(input_text)
        output_text = tokenizer.decode(tokenized.input_ids[0])
        assert input_text == output_text

    def test_tokenizer_batch_tokenizer(self):
        tokenizer = Tokenizer(tokenizer_name="hf-internal-testing/tiny-random-gpt2")
        input_text = ["Hello, world!", "Goodbye, world!"]
        tokenized = tokenizer.tokenize(input_text)
        output_text = tokenizer.batch_decode(tokenized.input_ids)
        assert input_text == output_text
