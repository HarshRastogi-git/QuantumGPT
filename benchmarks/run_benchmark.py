"""
benchmarks/run_benchmark.py — Detailed inference benchmark for QuantumGPT v2.
Profiles both models across multiple prompt lengths and temperatures.
"""
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe_tokenizer import BPETokenizer
from optimization.inference import InferenceProfiler, detect_device, tune_cpu_threads

TOKENIZER_PATH = "tokenizer/tokenizer.json"
CHECKPOINTS = {
    "baseline": "checkpoints/best_baseline.pkl",
    "gated": "checkpoints/best_gated.pkl",
}
PROMPTS = [
    "The",
    "To be or not to be",
    "It was a dark and stormy night and the detective",
    "She opened the letter and read the words carefully before",
]


def load_model_from_ckpt(path, device):
    import pickle
    from model.transformer import QuantumGPT, GPTConfig
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = GPTConfig(**ckpt["model_config"])
    model = QuantumGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    if cfg.use_gates:
        model.prune_heads()
    return model


def main():
    device_str = detect_device()
    tune_cpu_threads()
    device = torch.device(device_str)

    if not os.path.exists(TOKENIZER_PATH):
        print("Tokenizer not found. Run train.py first."); return

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    results = {}

    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"[benchmark] Skipping {name} — checkpoint not found"); continue

        print(f"\n── Benchmarking: {name} ──")
        model = load_model_from_ckpt(path, device)
        profiler = InferenceProfiler(model, tokenizer, device_str)

        model_results = {"prompts": {}}
        for prompt in PROMPTS:
            key = prompt[:30]
            print(f"  Prompt: '{key}...'")
            r = profiler.profile(prompt=prompt, n_tokens=100, n_runs=3)
            model_results["prompts"][key] = r
            print(f"    {r.get('mean_ms_per_token', '?')} ms/tok | {r.get('throughput_tok_per_sec', '?')} tok/s")

        results[name] = model_results

    out_path = "benchmarks/detailed_benchmark.json"
    os.makedirs("benchmarks", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed benchmark saved to {out_path}")


if __name__ == "__main__":
    main()
