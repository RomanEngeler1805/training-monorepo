import string


class BPETokenizer:
    def __init__(self):
        pass

    def pretokenize(self, text: str) -> dict:
        """
        Splits the given text into words

        inputs:
        - text: the text to derive the tokenization from

        outputs:
        - dict of words and counts
        """
        # TODO: seperate punctuation from words
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words_vocab: dict = {}
        for word in text.split(" "):
            if word in words_vocab:
                words_vocab[word] += 1
            else:
                words_vocab[word] = 1

        return words_vocab

    def create_symbols_vocab(self, words_vocab: dict) -> dict:
        """
        Split each word into its symbols; note, the sorting matters!
        """
        for key in words_vocab:
            words_vocab[key] = {"count": words_vocab[key], "vocab": list(key)}

        return words_vocab

    def learn_vocabulary(self, text: str, num_vocab: int):
        """
        Derive a vocabulary of size num_vocab from the text

        inputs:
        - text: text to derive the tokenization from
        - num_vocab: size of final vocabulary
        """
        # 1) Pre-tokenize
        words_vocab = self.pretokenize(text=text)

        # 2) Create base vocab of symbols
        symbols_vocab = self.create_symbols_vocab(words_vocab=words_vocab)

        # 3) Learn merge rules
        return symbols_vocab
