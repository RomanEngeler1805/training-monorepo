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
        words_vocab = tokenizer._pretokenize(text=text)

        assert words_vocab == expected_words_vocab

    def test_create_symbols_vocab(self):
        words_vocab = {"my": 3, "hugs": 2, "bun": 1}
        expected_symbols_vocab = {
            "my": {"count": 3, "vocab": ["m", "y"]},
            "hugs": {"count": 2, "vocab": ["h", "u", "g", "s"]},
            "bun": {"count": 1, "vocab": ["b", "u", "n"]},
        }

        tokenizer = BPETokenizer()
        symbols_vocab = tokenizer._create_symbols_vocab(words_vocab=words_vocab)

        assert symbols_vocab == expected_symbols_vocab

    def test_count_pairs(self):
        symbols_vocab = {
            "my": {"count": 3, "vocab": ["m", "y"]},
            "hugs": {"count": 2, "vocab": ["h", "u", "g", "s"]},
            "bun": {"count": 1, "vocab": ["b", "u", "n"]},
        }
        expected_pair_count = {
            ("m", "y"): 3,
            ("h", "u"): 2,
            ("u", "g"): 2,
            ("g", "s"): 2,
            ("b", "u"): 1,
            ("u", "n"): 1,
        }

        tokenizer = BPETokenizer()
        pair_count = tokenizer._count_pairs(vocab=symbols_vocab)

        assert pair_count == expected_pair_count

    def test_update_vocab(self):
        symbols_vocab = {
            "my": {"count": 3, "vocab": ["m", "y"]},
            "hugs": {"count": 2, "vocab": ["h", "u", "g", "s"]},
            "bun": {"count": 1, "vocab": ["b", "u", "n"]},
        }
        max_pair = ("m", "y")
        merged_token = "my"
        expected_vocab = {
            "my": {"count": 3, "vocab": ["my"]},
            "hugs": {"count": 2, "vocab": ["h", "u", "g", "s"]},
            "bun": {"count": 1, "vocab": ["b", "u", "n"]},
        }

        tokenizer = BPETokenizer()
        updated_vocab = tokenizer._update_vocab(
            max_pair=max_pair, merged_token=merged_token, vocab=symbols_vocab
        )

        assert updated_vocab == expected_vocab

    def test_merge_vocab(self):
        symbols_vocab = {
            "my": {"count": 3, "vocab": ["m", "y"]},
            "hugs": {"count": 2, "vocab": ["h", "u", "g", "s"]},
            "bun": {"count": 1, "vocab": ["b", "u", "n"]},
        }
        expected_full_vocab = {"m", "y", "h", "u", "g", "s", "b", "n", "my", "hu", "hug"}

        tokenizer = BPETokenizer()
        full_vocab = tokenizer._merge_vocab(vocab=symbols_vocab, num_vocab=11)
        print(full_vocab)

        assert full_vocab == expected_full_vocab
