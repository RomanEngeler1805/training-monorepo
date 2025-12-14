from models.bpe_tokenizer import BPETokenizer


class TestBPETokenizer:
    def test_pretokenization(self):
        text = "my vocabulary hugs my bun. i do like my hug. hugs are great!"
        expected_words_vocab = {
            "my": 3,
            "vocabulary": 1,
            "hugs": 2,
            "bun": 1,
            "i": 1,
            "do": 1,
            "like": 1,
            "hug": 1,
            "are": 1,
            "great": 1,
        }

        tokenizer = BPETokenizer()
        words_vocab = tokenizer.pretokenize(text=text)

        assert words_vocab == expected_words_vocab
