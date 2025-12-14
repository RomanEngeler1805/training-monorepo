import string
from typing import Any


class BPETokenizer:
    def __init__(self):
        pass

    def _pretokenize(self, text: str) -> dict[str, int]:
        """
        Split text into words and count their frequencies.

        Args:
            text: Input text to tokenize

        Returns:
            Dictionary mapping words to their counts
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words_vocab: dict[str, int] = {}
        for word in text.split(" "):
            if word in words_vocab:
                words_vocab[word] += 1
            else:
                words_vocab[word] = 1

        return words_vocab

    def _create_symbols_vocab(self, words_vocab: dict[str, int]) -> dict[str, dict[str, Any]]:
        """
        Convert word counts to symbol vocabularies (list of characters per word).

        Args:
            words_vocab: Dictionary mapping words to their counts

        Returns:
            Dictionary mapping words to {"count": int, "vocab": list[str]}
        """
        symbols_vocab: dict[str, dict[str, Any]] = {}
        for key, count in words_vocab.items():
            symbols_vocab[key] = {"count": count, "vocab": list(key)}

        return symbols_vocab

    def _count_pairs(self, vocab: dict[str, dict[str, Any]]) -> dict[tuple[str, str], int]:
        """
        Count frequency of adjacent token pairs across all words.

        Args:
            vocab: Dictionary mapping words to {"count": int, "vocab": list[str]}

        Returns:
            Dictionary mapping (token1, token2) pairs to their total counts
        """
        pair_count: dict[tuple[str, str], int] = {}
        for _, word_data in vocab.items():
            word_vocab = word_data["vocab"]
            count = word_data["count"]
            for i in range(len(word_vocab) - 1):
                pair = (word_vocab[i], word_vocab[i + 1])
                if pair in pair_count:
                    pair_count[pair] += count
                else:
                    pair_count[pair] = count
        return pair_count

    def _update_vocab(
        self, max_pair: tuple[str, str], merged_token: str, vocab: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Merge the most frequent pair in all word vocabularies.

        Args:
            max_pair: The (token1, token2) pair to merge
            merged_token: The new token created by merging the pair
            vocab: Dictionary mapping words to {"count": int, "vocab": list[str]}

        Returns:
            Updated vocab dictionary with merged tokens
        """
        for _, word_data in vocab.items():
            word_vocab = word_data["vocab"]
            i = 0
            while i < len(word_vocab) - 1:
                if (word_vocab[i], word_vocab[i + 1]) == max_pair:
                    word_vocab[i] = merged_token
                    word_vocab.pop(i + 1)
                else:
                    i += 1

        return vocab

    def _merge_vocab(self, vocab: dict[str, dict[str, Any]], num_vocab: int) -> set[str]:
        """
        Iteratively merge token pairs until vocabulary reaches target size.

        Args:
            vocab: Dictionary mapping words to {"count": int, "vocab": list[str]}
            num_vocab: Target vocabulary size

        Returns:
            Set of all unique tokens in the final vocabulary
        """
        # Build initial vocabulary set from all tokens
        full_vocab: set[str] = set()
        for key in vocab:
            full_vocab.update(vocab[key]["vocab"])

        while len(full_vocab) < num_vocab:
            pair_count = self._count_pairs(vocab=vocab)

            if not pair_count:
                break

            # Get the key for the max value
            max_pair = max(pair_count.items(), key=lambda x: x[1])[0]
            merged_token = max_pair[0] + max_pair[1]

            # Add merged token to full_vocab (individual chars stay in vocab)
            full_vocab.add(merged_token)

            # Update vocab by merging the pair
            vocab = self._update_vocab(max_pair=max_pair, merged_token=merged_token, vocab=vocab)

        return full_vocab

    def learn_vocabulary(self, text: str, num_vocab: int) -> set[str]:
        """
        Learn BPE vocabulary from text.

        Args:
            text: Input text to learn vocabulary from
            num_vocab: Target vocabulary size

        Returns:
            Set of all unique tokens in the learned vocabulary
        """
        # 1) Pre-tokenize
        words_vocab = self._pretokenize(text=text)

        # 2) Create base vocab of symbols
        symbols_vocab = self._create_symbols_vocab(words_vocab=words_vocab)

        # 3) Learn merge rules
        vocab = self._merge_vocab(vocab=symbols_vocab, num_vocab=num_vocab)

        return vocab
