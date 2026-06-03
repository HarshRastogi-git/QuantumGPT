"""
Dataset loader for QuantumGPT v2.
Handles tokenized corpus loading, train/val split, and efficient batch sampling.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class TextDataset(Dataset):
    """Token-level sliding-window dataset over a tokenized corpus."""

    def __init__(self, tokens: np.ndarray, block_size: int):
        self.tokens = tokens
        self.block_size = block_size
        # Number of complete windows
        self.n = len(tokens) - block_size

    def __len__(self) -> int:
        return max(0, self.n)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx: idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


class CorpusLoader:
    """
    Manages corpus tokenization, caching, and DataLoader creation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        block_size: int = 128,
        val_fraction: float = 0.1,
        cache_dir: str = "data",
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.val_fraction = val_fraction
        self.cache_dir = cache_dir
        self._tokens: Optional[np.ndarray] = None

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, "tokens.pkl")

    def load_or_tokenize(self, force: bool = False) -> np.ndarray:
        cache = self._cache_path()
        if not force and os.path.exists(cache):
            print(f"[CorpusLoader] Loading cached tokens from {cache}")
            with open(cache, "rb") as f:
                self._tokens = pickle.load(f)
            print(f"  ✓ {len(self._tokens):,} tokens loaded")
            return self._tokens

        print(f"[CorpusLoader] Tokenizing {self.data_path}...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize in chunks to show progress
        chunk_size = 50_000
        all_ids = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i: i + chunk_size]
            ids = self.tokenizer.encode(chunk)
            all_ids.extend(ids)
            if (i // chunk_size) % 5 == 0:
                pct = min(100, 100 * i // len(text))
                print(f"  Tokenizing... {pct}%")

        self._tokens = np.array(all_ids, dtype=np.int32)
        print(f"  ✓ {len(self._tokens):,} tokens")

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(self._tokens, f)
        print(f"  ✓ Cached to {cache}")
        return self._tokens

    def get_splits(self) -> Tuple[TextDataset, TextDataset]:
        """Return (train_dataset, val_dataset)."""
        if self._tokens is None:
            self.load_or_tokenize()
        n = len(self._tokens)
        split = int(n * (1 - self.val_fraction))
        train_tokens = self._tokens[:split]
        val_tokens = self._tokens[split:]
        print(f"[CorpusLoader] Train: {len(train_tokens):,} | Val: {len(val_tokens):,} tokens")
        return (
            TextDataset(train_tokens, self.block_size),
            TextDataset(val_tokens, self.block_size),
        )

    def get_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds, val_ds = self.get_splits()
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        return train_loader, val_loader
