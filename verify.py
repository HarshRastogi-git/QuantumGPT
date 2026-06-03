"""
verify.py — Confirm the gated model generates coherently after pruning.
Run this BEFORE starting the UI server to sanity-check your checkpoints.

    python verify.py
    python verify.py --gated_path checkpoints/gated_v5.pkl

Prints:
  - Gate values per layer (pre-pruning)
  - Pruning decisions
  - Sample generation from the BROKEN path (no pruning, eval mode)
  - Sample generation from the FIXED path (pruned, same as UI will use)
"""

import argparse
import os
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.transformer import GPTConfig, QuantumGPT
from tokenizer.bpe_tokenizer import BPETokenizer

TOKENIZER_PATH = "tokenizer/tokenizer.json"

_GATE_DEFAULTS = {
    "use_gates": False, "gate_reg_lambda": 0.0, "gate_threshold": 0.2,
    "gate_prune_pct": 0.33, "gate_reg_start": 0.3,
    "gate_temp_start": 4.0, "gate_temp_end": 10.0, "gate_binaryness": 0.5,
}

PROMPTS = [
    "To be, or not to be, that is the question:",
    "Friends, Romans, countrymen, lend me your ears;",
    "It was the best of times, it was the worst of times,",
]


def load_raw(path: str) -> QuantumGPT:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    cfg_dict = ckpt["model_config"]
    for k, v in _GATE_DEFAULTS.items():
        cfg_dict.setdefault(k, v)
    valid = set(GPTConfig.__dataclass_fields__)
    for stale in [k for k in list(cfg_dict) if k not in valid]:
        cfg_dict.pop(stale)
    config = GPTConfig(**cfg_dict)
    model  = QuantumGPT(config)
    model.load_state_dict(ckpt["model_state"])
    return model


def generate_clean(model, tokenizer, prompt, n=80, temperature=0.8, top_k=50):
    """Clean generate that doesn't use model.generate() or depend on training flags."""
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        return "(empty)"
    idx = torch.tensor([ids], dtype=torch.long)
    out = list(ids)
    with torch.no_grad():
        for _ in range(n):
            cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                thresh = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt   = int(torch.multinomial(probs, 1).item())
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return tokenizer.decode(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gated_path",    default="checkpoints/gated_v5.pkl")
    p.add_argument("--baseline_path", default="checkpoints/baseline.pkl")
    p.add_argument("--temperature",   type=float, default=0.8)
    p.add_argument("--top_k",         type=int,   default=50)
    p.add_argument("--n_tokens",      type=int,   default=80)
    return p.parse_args()


def main():
    args      = parse_args()
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    W = 66

    # ── Baseline ─────────────────────────────────────────────────────────
    if os.path.exists(args.baseline_path):
        print(f"\n{'═'*W}")
        print(f"  BASELINE — {args.baseline_path}")
        print(f"{'═'*W}")
        model = load_raw(args.baseline_path)
        model.eval()
        print(f"  {model}")
        print(f"  Size: {model.model_size_mb():.3f} MB  |  Params: {model.num_parameters():,}")
        for prompt in PROMPTS:
            print(f"\n  Prompt: {prompt}")
            print(f"  Output: {generate_clean(model, tokenizer, prompt, args.n_tokens, args.temperature, args.top_k)}")

    # ── Gated ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.gated_path):
        print(f"\n[SKIP] {args.gated_path} not found")
        return

    print(f"\n{'═'*W}")
    print(f"  GATED (raw checkpoint) — {args.gated_path}")
    print(f"{'═'*W}")

    model = load_raw(args.gated_path)

    # ── Gate analysis ──────────────────────────────────────────────────────
    if model.config.use_gates:
        print("\n  Gate values per layer (using gate_temp_end for clarity):")
        final_temp = model.config.gate_temp_end
        threshold  = model.config.gate_threshold
        for i, blk in enumerate(model.transformer["h"]):
            blk.attn.set_temp(final_temp)
            with torch.no_grad():
                gates = torch.sigmoid(blk.attn.g_raw * final_temp).tolist()
            bar = "".join(
                "█" if g >= threshold else "░" for g in gates
            )
            vals = "  ".join(f"{g:.3f}" for g in gates)
            n_live = sum(1 for g in gates if g >= threshold)
            print(f"    Layer {i}: [{bar}]  {vals}  ({n_live}/{len(gates)} live)")

    print(f"\n  {model}")

    # ── BROKEN path: eval mode, no pruning ────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  BROKEN PATH — eval mode, gate not applied (what the old UI did):")
    print(f"  {'─'*W}")
    model.eval()   # self.training=False → gate condition fails → mismatch
    for prompt in PROMPTS[:1]:
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {generate_clean(model, tokenizer, prompt, args.n_tokens, args.temperature, args.top_k)}")

    # ── FIXED path: prune first, then generate ────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  FIXED PATH — prune_heads() + structurally_prune() first:")
    print(f"  {'─'*W}")

    # Reload fresh (so we're not using the already-eval'd version)
    model = load_raw(args.gated_path)

    if model.config.use_gates:
        final_temp = model.config.gate_temp_end
        for blk in model.transformer["h"]:
            blk.attn.set_temp(final_temp)
        prune_stats = model.prune_heads()
        model.structurally_prune()
        model.eval()

        print(f"\n  Prune results:")
        print(f"    Total heads:  {prune_stats['total_heads']}")
        print(f"    Pruned heads: {prune_stats['pruned_heads']}  ({prune_stats['prune_pct']:.1f}%)")
        print(f"    Live heads:   {prune_stats['active_heads']}")
        for row in prune_stats["by_layer"]:
            status = "active" * row["active"] + "  dead" * row["pruned"]
            print(f"    Layer {row['layer']}: {row['active']}/{row['active']+row['pruned']} live")

    print(f"\n  {model}")
    print(f"  Size after pruning: {model.model_size_mb():.3f} MB  |  Params: {model.num_parameters():,}")

    for prompt in PROMPTS:
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {generate_clean(model, tokenizer, prompt, args.n_tokens, args.temperature, args.top_k)}")

    print(f"\n{'═'*W}")
    print(f"  If the FIXED PATH output looks coherent, the UI server will work.")
    print(f"  The server does the same prune_heads() + structurally_prune()")
    print(f"  automatically whenever it loads a gated checkpoint.")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()