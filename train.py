"""
train.py — Train the BASELINE QuantumGPT model (no gating).
Usage:
    python train.py [--iters N] [--batch B] [--device cpu|cuda]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import QuantumGPT, GPTConfig
from training.trainer import Trainer, TrainConfig
from training.dataset import CorpusLoader

TOKENIZER_PATH = "tokenizer/tokenizer.json"
DATA_PATH = "data/raw.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline QuantumGPT")
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--retokenize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Tokenizer ──────────────────────────────────────────────────
    if os.path.exists(TOKENIZER_PATH):
        print("[train.py] Loading existing tokenizer...")
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    else:
        print("[train.py] Training tokenizer from scratch...")
        if not os.path.exists(DATA_PATH):
            print(f"ERROR: {DATA_PATH} not found. Run: python data/prepare_data.py")
            sys.exit(1)
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = BPETokenizer(vocab_size=4000)
        tokenizer.train(text)
        tokenizer.save(TOKENIZER_PATH)

    # ── 2. Dataset ────────────────────────────────────────────────────
    model_cfg = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=128,
        n_layer=4,
        n_head=6,
        n_embd=192,
        dropout=0.1,
        use_gates=False,
    )

    loader = CorpusLoader(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        block_size=model_cfg.block_size,
        cache_dir="data",
    )
    if args.retokenize:
        loader.load_or_tokenize(force=True)

    train_loader, val_loader = loader.get_loaders(batch_size=args.batch)

    # ── 3. Model ──────────────────────────────────────────────────────
    model = QuantumGPT(model_cfg)
    print(f"\n{model}")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Size: {model.model_size_mb():.2f} MB\n")

    # ── 4. Train ──────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        checkpoint_dir="checkpoints",
        checkpoint_name="baseline.pkl",
        log_dir="logs",
        batch_size=args.batch,
        learning_rate=3e-4,
        max_iters=args.iters,
        warmup_iters=max(100, args.iters // 25),
        eval_interval=max(200, args.iters // 25),
        eval_iters=50,
        device=args.device,
    )

    trainer = Trainer(model, train_loader, val_loader, train_cfg)
    history = trainer.train()

    print("\n✓ Baseline training complete!")
    print(f"  Checkpoint: checkpoints/baseline.pkl")
    print(f"  Best val loss: {history['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
