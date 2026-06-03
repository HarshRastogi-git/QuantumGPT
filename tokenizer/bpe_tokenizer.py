"""
BPE (Byte-Pair Encoding) Tokenizer — implemented from scratch.
Trains on raw text, saves/loads as JSON.
"""
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch.
    Supports encode/decode, special tokens, and JSON serialization.
    """

    # Special tokens
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN
        ]
        self._trained = False

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def _get_word_freqs(self, text: str) -> Dict[Tuple[str, ...], int]:
        """Tokenize text into character-level words with frequency counts."""
        # Simple whitespace/punctuation splitter — keeps punctuation as own tokens
        pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]|\s+(?!\S)|\s""")
        words = re.findall(pattern, text)
        freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        for word in words:
            chars = tuple(list(word) + ["</w>"])
            freq[chars] += 1
        return freq

    def _get_pair_stats(self, vocab_freq: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of every adjacent symbol pair."""
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, freq in vocab_freq.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: Tuple[str, str],
        vocab_freq: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of a pair in the vocabulary."""
        new_vocab: Dict[Tuple[str, ...], int] = {}
        bigram = pair
        replacement = pair[0] + pair[1]
        for word, freq in vocab_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(self, text: str, verbose: bool = True) -> None:
        """Train BPE on the given text corpus."""
        if verbose:
            print(f"[BPETokenizer] Training on {len(text):,} chars, target vocab={self.vocab_size}")

        # Start with character vocabulary
        all_chars: set = set()
        for ch in text:
            if 0x20 <= ord(ch) <= 0x7E or ch == "\n":
                all_chars.add(ch)
        all_chars.add("</w>")

        # Build initial vocab
        base_vocab = sorted(all_chars)
        self.vocab = {}

        # Special tokens first
        for tok in self.special_tokens:
            self.vocab[tok] = len(self.vocab)

        for ch in base_vocab:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        self.merges = []
        word_freqs = self._get_word_freqs(text)

        num_merges = self.vocab_size - len(self.vocab)
        if verbose:
            print(f"  Base vocab size: {len(self.vocab)}")
            print(f"  Learning {num_merges} merges...")

        log_every = max(1, num_merges // 10)

        for i in range(num_merges):
            pairs = self._get_pair_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_vocab(best_pair, word_freqs)
            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
            if verbose and (i + 1) % log_every == 0:
                print(f"  Merge {i+1}/{num_merges} | vocab size: {len(self.vocab)}")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True
        if verbose:
            print(f"  ✓ Final vocab size: {len(self.vocab)}")

    # ------------------------------------------------------------------ #
    #  Encoding / Decoding                                                 #
    # ------------------------------------------------------------------ #

    def _bpe_word(self, word: Tuple[str, ...]) -> List[str]:
        """Apply learned BPE merges to a single word."""
        word_list = list(word)
        for merge in self.merges:
            new_word = []
            i = 0
            while i < len(word_list):
                if i < len(word_list) - 1 and (word_list[i], word_list[i + 1]) == merge:
                    new_word.append(merge[0] + merge[1])
                    i += 2
                else:
                    new_word.append(word_list[i])
                    i += 1
            word_list = new_word
        return word_list

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Encode a string to token IDs."""
        assert self._trained, "Tokenizer not trained. Call .train() or .load() first."
        pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]|\s+(?!\S)|\s""")
        words = re.findall(pattern, text)
        token_ids = []
        unk_id = self.vocab.get(self.UNK_TOKEN, 0)
        if add_special:
            token_ids.append(self.vocab[self.BOS_TOKEN])
        for word in words:
            chars = tuple(list(word) + ["</w>"])
            subwords = self._bpe_word(chars)
            for sw in subwords:
                token_ids.append(self.vocab.get(sw, unk_id))
        if add_special:
            token_ids.append(self.vocab[self.EOS_TOKEN])
        return token_ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to string."""
        assert self._trained
        special_set = set(self.special_tokens)
        tokens = []
        for tid in token_ids:
            tok = self.inverse_vocab.get(tid, self.UNK_TOKEN)
            if skip_special and tok in special_set:
                continue
            tokens.append(tok)
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()
        # return text

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    #  Serialization                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save tokenizer to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BPETokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.special_tokens = data["special_tokens"]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok._trained = True
        print(f"[BPETokenizer] Loaded from {path} | vocab={len(tok.vocab)}")
        return tok
