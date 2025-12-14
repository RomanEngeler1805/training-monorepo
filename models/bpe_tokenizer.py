import string


class BPETokenizer:
    def __init__(self):
        pass

    def pretokenize(self, text: str):
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
        # 3) Learn merge rules
        return words_vocab
