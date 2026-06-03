"""
evaluate.py — Ablation study: Baseline vs Gated model (QuantumGPT v2).
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from typing import Dict, Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import QuantumGPT, GPTConfig
from training.dataset import CorpusLoader

TOKENIZER_PATH = "tokenizer/tokenizer.json"
DATA_PATH      = "data/raw.txt"
RESULTS_PATH   = "benchmarks/ablation_results.json"

SAMPLE_PROMPTS = [
    "To be, or not to be, that is the question",
    "It was the best of times, it was the worst of times",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_path", type=str, default="checkpoints/baseline.pkl")
    p.add_argument("--gated_path",    type=str, default="checkpoints/gated_v5.pkl")
    p.add_argument("--device",        type=str,  default="cpu")
    p.add_argument("--eval_iters",    type=int,  default=100)
    p.add_argument("--gen_tokens",    type=int,  default=200)
    p.add_argument("--gen_runs",      type=int,  default=3)
    p.add_argument("--sample_tokens", type=int,  default=80)
    return p.parse_args()


def load_model_from_ckpt(path: str, device: str) -> QuantumGPT:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    cfg = ckpt["model_config"]
    gate_defaults = {
        "use_gates": False, "gate_reg_lambda": 0.0, "gate_threshold": 0.2,
        "gate_prune_pct": 0.33, "gate_reg_start": 0.3, "gate_temp_start": 4.0,
        "gate_temp_end": 10.0, "gate_binaryness": 0.5,
    }
    for k, v in gate_defaults.items():
        cfg.setdefault(k, v)

    valid_fields = {f.name for f in GPTConfig.__dataclass_fields__.values()}
    stale = [k for k in list(cfg.keys()) if k not in valid_fields]
    for k in stale:
        cfg.pop(k)

    config = GPTConfig(**cfg)
    model  = QuantumGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model

@torch.no_grad()
def compute_perplexity(model, val_loader, device, max_iters=100) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_iters: break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        count += 1
    return math.exp(min(total / max(count, 1), 20))

def measure_latency(model, tokenizer, device, n_tokens=200, n_runs=3) -> Dict:
    model.eval()
    ids    = tokenizer.encode("To be or not to be, that is the question")
    prompt = torch.tensor([ids], dtype=torch.long, device=device)

    all_ms = []
    for _ in range(n_runs):
        for _, ms in model.generate(prompt.clone(), max_new_tokens=n_tokens, temperature=0.8):
            all_ms.append(ms)

    if not all_ms:
        return {"mean_ms_per_token": 0, "throughput_tok_per_sec": 0, "p50_ms": 0, "p95_ms": 0}

    s   = sorted(all_ms)
    n   = len(s)
    avg = sum(s) / n
    return {
        "mean_ms_per_token":      round(avg, 3),
        "p50_ms":                 round(s[n // 2], 3),
        "p95_ms":                 round(s[min(int(n * 0.95), n-1)], 3),
        "throughput_tok_per_sec": round(1000 / max(avg, 1e-6), 2),
    }

def generate_sample(model, tokenizer, device, prompt: str, n_tokens=80) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids: return ""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generated = list(ids)
    with torch.no_grad():
        for tok_id, _ in model.generate(idx, max_new_tokens=n_tokens, temperature=0.85, top_k=50):
            generated.append(tok_id)
    return tokenizer.decode(generated)


def evaluate_model(name, path, tokenizer, val_loader, device, args) -> Dict:
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: not found at {path}")
        return {}

    print(f"\n{'─'*56}")
    print(f"  {name} | Checkpoint : {path}")

    model = load_model_from_ckpt(path, device)
    
    # --- PRUNING LOGIC ---
    prune_stats = {}
    if model.config.use_gates:
        print(f"\n  Analyzing Heads...")
        prune_stats = model.prune_heads()
        
        # ACTUALLY COMPRESS THE MODEL
        print(f"  Applying Structural Pruning (Compressing Weights)...")
        model.structurally_prune()
    else:
        prune_stats = {
            "total_heads": model.config.n_layer * model.config.n_head,
            "active_heads": model.config.n_layer * model.config.n_head,
            "pruned_heads": 0,
            "prune_pct": 0.0
        }

    print(f"  Model      : {model}")
    print(f"  Size       : {model.model_size_mb():.2f} MB | Params: {model.num_parameters():,}")

    print(f"\n  Perplexity ({args.eval_iters} batches)...")
    ppl = compute_perplexity(model, val_loader, device, args.eval_iters)
    print(f"  → {ppl:.3f}")

    print(f"\n  Latency ({args.gen_tokens} tokens × {args.gen_runs} runs)...")
    lat = measure_latency(model, tokenizer, device, args.gen_tokens, args.gen_runs)
    print(f"  → {lat['mean_ms_per_token']} ms/tok | {lat['throughput_tok_per_sec']} tok/s")

    samples = {}
    print(f"\n  Sample generation:")
    for prompt in SAMPLE_PROMPTS:
        text = generate_sample(model, tokenizer, device, prompt, args.sample_tokens)
        samples[prompt] = text

    return {
        "name":                   name,
        "perplexity":             round(ppl, 3),
        "latency_ms_per_token":   lat["mean_ms_per_token"],
        "throughput_tok_per_sec": lat["throughput_tok_per_sec"],
        "p50_ms":                 lat["p50_ms"],
        "p95_ms":                 lat["p95_ms"],
        "model_size_mb":          round(model.model_size_mb(), 3),
        "num_parameters":         model.num_parameters(),
        "total_heads":            prune_stats.get("total_heads", 0),
        "active_heads":           prune_stats.get("active_heads", 0),
        "pruned_heads":           prune_stats.get("pruned_heads", 0),
        "prune_pct":              prune_stats.get("prune_pct", 0.0),
        "samples":                samples,
    }


def print_comparison_table(baseline: Dict, gated: Dict) -> None:
    if not baseline or not gated: return

    def change(key, lower_better=True):
        b, g = baseline.get(key), gated.get(key)
        if b is None or g is None or b == 0: return "—"
        pct = (g - b) / abs(b) * 100
        if abs(pct) < 0.05: return "±0.0%"
        arrow = "↓" if pct < 0 else "↑"
        better = (pct < 0) == lower_better
        tag = " ✓" if better else "  "
        return f"{'+'if pct>0 else ''}{pct:.1f}%{tag}{arrow}"

    def fmt(v):
        if isinstance(v, int):   return f"{v:,}"
        if isinstance(v, float): return f"{v}"
        return str(v) if v is not None else "—"

    rows = [
        ("Perplexity (↓ better)",      "perplexity",             True),
        ("Latency ms/tok (↓ better)",  "latency_ms_per_token",   True),
        ("Throughput tok/s (↑ better)","throughput_tok_per_sec", False),
        ("Model size MB (↓ better)",   "model_size_mb",          True),
        ("Parameters (↓ better)",      "num_parameters",         True),
        ("Active heads",               "active_heads",           True),
    ]

    W = 76
    print(f"\n{'═'*W}")
    print(f"  ABLATION STUDY — QuantumGPT v2  (Structural Pruning)")
    print(f"{'═'*W}")
    print(f"  {'Metric':<30} {'Baseline':>12} {'Gated':>12} {'Change':>18}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*18}")
    for label, key, lb in rows:
        bv, gv = fmt(baseline.get(key)), fmt(gated.get(key))
        print(f"  {label:<30} {bv:>12} {gv:>12} {change(key, lb):>18}")
    print(f"{'═'*W}\n")

def main():
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs("benchmarks", exist_ok=True)

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    loader    = CorpusLoader(DATA_PATH, tokenizer, block_size=128, cache_dir="data")
    _, val_loader = loader.get_loaders(batch_size=32)

    baseline = evaluate_model("Baseline", args.baseline_path, tokenizer, val_loader, device, args)
    gated    = evaluate_model("Gated (Structurally Pruned)", args.gated_path, tokenizer, val_loader, device, args)

    print_comparison_table(baseline, gated)

if __name__ == "__main__":
    main()

