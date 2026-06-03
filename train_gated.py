"""
train_gated.py — Train the GATED QuantumGPT (Adaptive Gated Head Pruning).

Design:
  - Raw sigmoid gates (NO normalization) so gates can truly go to 0 or 1
  - Small random init (0.05*randn) breaks symmetry early
  - Temperature ramp: sigmoid gets steeper during phase 2 (sharper separation)
  - Loss: sparsity + binaryness (pushes toward 0 OR 1, not stuck at 0.5)
  - Pruning: rank-based (lowest 33% per layer) union fixed threshold (0.2)
  - Saves to gated_v4.pkl — all existing checkpoints untouched

Usage:
    python train_gated.py --warmstart
    python train_gated.py --iters 5000 --warmstart --gate_lambda 1e-2
"""
import argparse
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import QuantumGPT, GPTConfig
from training.trainer import Trainer, TrainConfig
from training.dataset import CorpusLoader

TOKENIZER_PATH = "tokenizer/tokenizer.json"
DATA_PATH      = "data/raw.txt"
BASELINE_PATH  = "checkpoints/baseline.pkl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters",            type=int,   default=5000)
    p.add_argument("--batch",            type=int,   default=32)
    p.add_argument("--device",           type=str,   default="cpu")
    p.add_argument("--gate_lambda",      type=float, default=5e-2,
                   help="Binaryness loss weight. Higher = faster bifurcation (default 5e-2)")
    p.add_argument("--gate_threshold",   type=float, default=0.2,
                   help="Fixed threshold on raw sigmoid gate (0-1). Default 0.2")
    p.add_argument("--gate_prune_pct",   type=float, default=0.33,
                   help="Rank-based: prune lowest fraction per layer. Default 0.33")
    p.add_argument("--gate_reg_start",   type=float, default=0.25,
                   help="Fraction of training before reg begins (default 0.25). Earlier = higher LR during phase 2")
    p.add_argument("--gate_temp_start",  type=float, default=4.0,
                   help="Sigmoid temp at start of phase 2. Default 4.0")
    p.add_argument("--gate_temp_end",    type=float, default=10.0,
                   help="Sigmoid temp at end of training. Default 10.0")
    p.add_argument("--gate_binaryness",  type=float, default=0.5,
                   help="Weight of binaryness term in reg loss. Default 0.5")
    p.add_argument("--gate_lr",          type=float, default=5e-2,
                   help="Fixed LR for gate params (no cosine decay, default 5e-2). Higher needed for binaryness-only loss")
    p.add_argument("--warmstart",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Tokenizer ──────────────────────────────────────────────────────────
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: {TOKENIZER_PATH} not found. Run train.py first.")
        sys.exit(1)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    # ── Config ─────────────────────────────────────────────────────────────
    model_cfg = GPTConfig(
        vocab_size       = len(tokenizer),
        block_size       = 128,
        n_layer          = 4,
        n_head           = 6,
        n_embd           = 192,
        dropout          = 0.1,
        use_gates        = True,
        gate_reg_lambda  = args.gate_lambda,
        gate_threshold   = args.gate_threshold,
        gate_prune_pct   = args.gate_prune_pct,
        gate_reg_start   = args.gate_reg_start,
        gate_temp_start  = args.gate_temp_start,
        gate_temp_end    = args.gate_temp_end,
        gate_binaryness  = args.gate_binaryness,
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    loader = CorpusLoader(
        data_path  = DATA_PATH,
        tokenizer  = tokenizer,
        block_size = model_cfg.block_size,
        cache_dir  = "data",
    )
    train_loader, val_loader = loader.get_loaders(batch_size=args.batch)

    # ── Model ──────────────────────────────────────────────────────────────
    model = QuantumGPT(model_cfg)

    if args.warmstart and os.path.exists(BASELINE_PATH):
        print(f"[train_gated] Warm-starting from {BASELINE_PATH}...")
        with open(BASELINE_PATH, "rb") as f:
            ckpt = pickle.load(f)
        model_state = model.state_dict()
        loaded = 0
        for k, v in ckpt["model_state"].items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  ✓ Loaded {loaded} weight tensors (g_raw stays random init)")
    else:
        print("[train_gated] Training from scratch.")

    phase2_iter = int(args.gate_reg_start * args.iters)
    print(f"\n{model}")
    print(f"  Params           : {model.num_parameters():,}")
    print(f"  gate_lambda      : {args.gate_lambda}  (single scaling)")
    print(f"  gate_binaryness  : {args.gate_binaryness}")
    print(f"  gate_lr          : {args.gate_lr}  (fixed, no decay)")
    print(f"  temp ramp        : {args.gate_temp_start} → {args.gate_temp_end}  (phase 2)")
    print(f"  Phase 2 starts   : iter {phase2_iter}/{args.iters}  ({int(args.gate_reg_start*100)}%)")
    print(f"  Pruning          : rank bottom {int(args.gate_prune_pct*100)}%/layer  OR  gate < {args.gate_threshold}")
    print()

    # ── Gradient check (with reg active) ───────────────────────────────────
    print("  Gradient check...")
    import torch as _t
    _x  = _t.randint(0, len(tokenizer), (2, model_cfg.block_size))
    _y  = _t.randint(0, len(tokenizer), (2, model_cfg.block_size))
    _ps = int(args.gate_reg_start * args.iters)
    _, _loss = model(_x, _y, current_iter=_ps, max_iter=args.iters)
    _loss.backward()
    _ok = True
    for _i, _blk in enumerate(model.transformer["h"]):
        _g = _blk.attn.g_raw
        if _g is None or _g.grad is None or _g.grad.abs().max() < 1e-10:
            print(f"    layer_{_i}: *** GRADIENT PROBLEM ***")
            _ok = False
        else:
            print(f"    layer_{_i}: grad_max={_g.grad.abs().max().item():.5f}  ✓")
    model.zero_grad()
    if not _ok:
        print("  GRADIENT CHECK FAILED — aborting.")
        sys.exit(1)
    print("  ✓ Gradients confirmed\n")

    # ── Initial gate display ───────────────────────────────────────────────
    print("  Initial raw gates (small random noise around 0.5):")
    for layer, gates in model.get_gate_values().items():
        print(f"    {layer}: [{' '.join(f'{g:.3f}' for g in gates)}]")
    print()

    # ── Train ──────────────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        checkpoint_dir  = "checkpoints",
        checkpoint_name = "gated_v5.pkl",
        log_dir         = "logs",
        batch_size      = args.batch,
        learning_rate   = 3e-4,
        max_iters       = args.iters,
        warmup_iters    = max(100, args.iters // 25),
        eval_interval   = max(200, args.iters // 25),
        eval_iters      = 50,
        device          = args.device,
        gate_lr         = args.gate_lr,
    )

    trainer = Trainer(model, train_loader, val_loader, train_cfg)
    history = trainer.train()

    # ── Post-training analysis ─────────────────────────────────────────────
    print("\n── Post-Training Gate Analysis (raw sigmoid, 0=dead 1=active) ──")
    gate_vals = model.get_gate_values()
    for layer, gates in gate_vals.items():
        gstr  = " ".join(f"{g:.3f}" for g in gates)
        flags = " ".join("✗" if g < args.gate_threshold else "✓" for g in gates)
        bar   = " ".join("█" if g > 0.5 else "░" for g in gates)
        print(f"  {layer}: [{gstr}]  [{bar}]  {flags}")

    prune_stats = model.prune_heads()
    print(f"\n  Total:  {prune_stats['total_heads']}")
    print(f"  Active: {prune_stats['active_heads']}")
    print(f"  Pruned: {prune_stats['pruned_heads']}  ({prune_stats['prune_pct']:.1f}%)")
    for li in prune_stats["by_layer"]:
        bar = "█" * li["active"] + "░" * li["pruned"]
        print(f"    layer_{li['layer']}: [{bar}]  {li['active']}/{li['active']+li['pruned']} active")

    # Save with pruning metadata
    out_path = "checkpoints/gated_v5.pkl"
    with open(out_path, "rb") as f:
        ckpt = pickle.load(f)
    ckpt["prune_stats"] = prune_stats
    ckpt["gate_values"] = gate_vals
    with open(out_path, "wb") as f:
        pickle.dump(ckpt, f)

    print(f"\n✓ Done  |  checkpoint: {out_path}")
    print(f"  Best val loss : {history['best_val_loss']:.4f}")
    print(f"  Heads pruned  : {prune_stats['prune_pct']:.1f}%")


if __name__ == "__main__":
    main()