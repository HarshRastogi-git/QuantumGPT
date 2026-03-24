import re

class SimpleTokenizer:
    def tokenize(self, text):
        # Tokenize by splitting on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return tokens

class BPE:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.vocab = {}

    def train(self, corpus):
        # Count frequency of pairs of characters
        for word in corpus:
            symbols = list(word)
            self.vocab[word] = self.vocab.get(word, 0) + 1

        for _ in range(self.num_merges):
            # Find the most frequent pair
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair)

    def get_stats(self):
        pairs = {}
        for word, freq in self.vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def merge_vocab(self, pair):
        new_vocab = {}
        bigram = ''.join(pair)
        replacement = ''.join(pair[0]) + ''.join(pair[1])
        for word in self.vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab

    def encode(self, text):
        # Encode using BPE
        tokens = text.split()
        return [self.vocab.get(token, token) for token in tokens]

    def decode(self, tokens):
        # Decode BPE tokens to original text
        return ' '.join(tokens)
