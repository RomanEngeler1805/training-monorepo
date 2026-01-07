import pytest
import torch

from src.models.hf_model import HFModel


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_length():
    return 5


@pytest.fixture
def n_vocab():
    return 524288


@pytest.fixture
def input_ids(batch_size, seq_length, n_vocab):
    return torch.randint(low=0, high=n_vocab, size=(batch_size, seq_length))


@pytest.fixture
def attention_mask(batch_size, seq_length):
    return torch.ones(batch_size, seq_length, dtype=torch.long)


# Model fixtures (HuggingFace model)
@pytest.fixture
def model_name():
    return "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture
def hf_model(model_name):
    return HFModel(model_name=model_name)


class TestHFModel:
    def test_init(self, hf_model):
        assert hf_model is not None
        assert hf_model.model is not None

    def test_model_parameters(self, hf_model):
        params = list(hf_model.parameters())
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) or hasattr(p, "requires_grad") for p in params)

    def test_model_forward(self, hf_model, input_ids, attention_mask, batch_size, seq_length):
        output = hf_model.forward(input_ids, attention_mask)
        n_vocab = hf_model.model.config.vocab_size

        assert output.logits.shape == (batch_size, seq_length, n_vocab)
        assert output.logits.dtype == torch.bfloat16
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()

        assert not torch.isnan(output.logits).any()

    def test_model_train(self, hf_model):
        hf_model.train()
        assert hf_model.model.training

    def test_model_eval(self, hf_model):
        hf_model.eval()
        assert not hf_model.model.training
