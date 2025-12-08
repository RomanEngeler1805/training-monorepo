import torch

from inference.decoding import GreedyDecoder


class TestGreedyDecoder:
    def test_decode(self):
        input_ids = torch.randint(low=0, high=256, size=(8, 5))
        logits = torch.rand(8, 256)

        decoder = GreedyDecoder()
        token_ids = decoder.decode(input_ids=input_ids, logits=logits)

        assert token_ids.shape == (8, 6)
