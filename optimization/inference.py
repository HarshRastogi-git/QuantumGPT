"""
QuantumGPT v2 — Optimization Utilities

Covers:
  - Device detection (CPU / CUDA / AMD ROCm)
  - FP16 half-precision inference
  - Torch compile (optional)
  - Inference profiler
"""
import time
import platform
import torch
import torch.nn as nn
from typing import Optional
from model.transformer import QuantumGPT


# ─────────────────────────────────────────────────────────────────────────── #
#  Device Detection                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def detect_device() -> str:
    """
    Returns best available device string.
    AMD ROCm exposes itself as 'cuda' in PyTorch.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # ROCm / AMD detection
        if "AMD" in name or "Radeon" in name or "gfx" in name.lower():
            print(f"[optimize] AMD GPU detected: {name} (via ROCm/HIP)")
        else:
            print(f"[optimize] CUDA GPU detected: {name}")
        return "cuda"
    print(f"[optimize] No GPU available — using CPU ({platform.processor()})")
    return "cpu"


def device_info() -> dict:
    info = {"device": detect_device(), "platform": platform.platform()}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda or "ROCm"
    info["torch_version"] = torch.__version__
    info["cpu_threads"] = torch.get_num_threads()
    return info


# ─────────────────────────────────────────────────────────────────────────── #
#  FP16 Quantization                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def to_half_precision(model: QuantumGPT, device: str = "cuda") -> QuantumGPT:
    """
    Convert model to FP16 for faster GPU inference.
    Only meaningful on GPU — CPU FP16 is usually slower.
    """
    if device == "cpu":
        print("[optimize] FP16 skipped — CPU doesn't benefit from half precision")
        return model
    model = model.half()
    print("[optimize] Model converted to FP16")
    return model


def restore_full_precision(model: QuantumGPT) -> QuantumGPT:
    return model.float()


# ─────────────────────────────────────────────────────────────────────────── #
#  Torch Compile                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def try_compile(model: QuantumGPT) -> QuantumGPT:
    """
    Attempt torch.compile for ~20-30% speedup (PyTorch 2.0+).
    Falls back gracefully if not supported.
    """
    if not hasattr(torch, "compile"):
        print("[optimize] torch.compile not available (requires PyTorch 2.0+)")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("[optimize] torch.compile applied (mode=reduce-overhead)")
        return compiled
    except Exception as e:
        print(f"[optimize] torch.compile failed: {e} — using eager mode")
        return model


# ─────────────────────────────────────────────────────────────────────────── #
#  CPU Thread Tuning                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def tune_cpu_threads(n_threads: Optional[int] = None) -> int:
    """Set PyTorch inter-op threads for CPU inference."""
    import os
    if n_threads is None:
        # Use physical core count, cap at 8
        n_threads = min(os.cpu_count() or 4, 8)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))
    print(f"[optimize] CPU threads: {n_threads} intra / {max(1, n_threads//2)} inter")
    return n_threads


# ─────────────────────────────────────────────────────────────────────────── #
#  Inference Profiler                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class InferenceProfiler:
    """
    Profiles a model's generation speed.
    Returns tokens/sec, latency percentiles, and memory usage.
    """

    def __init__(self, model: QuantumGPT, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @torch.no_grad()
    def profile(
        self,
        prompt: str = "The",
        n_tokens: int = 100,
        n_runs: int = 3,
        temperature: float = 0.8,
    ) -> dict:
        self.model.eval()
        token_ids = self.tokenizer.encode(prompt) or [0]
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        all_latencies = []
        for run in range(n_runs):
            ctx = idx.clone()
            for tok_id, ms in self.model.generate(ctx, max_new_tokens=n_tokens, temperature=temperature):
                all_latencies.append(ms)

        if not all_latencies:
            return {}

        sorted_lat = sorted(all_latencies)
        n = len(sorted_lat)
        mean_ms = sum(sorted_lat) / n
        p50 = sorted_lat[n // 2]
        p90 = sorted_lat[int(n * 0.9)]
        p99 = sorted_lat[min(int(n * 0.99), n - 1)]
        throughput = 1000.0 / mean_ms

        result = {
            "mean_ms_per_token": round(mean_ms, 3),
            "p50_ms": round(p50, 3),
            "p90_ms": round(p90, 3),
            "p99_ms": round(p99, 3),
            "throughput_tok_per_sec": round(throughput, 2),
            "total_tokens_measured": n,
            "n_runs": n_runs,
        }

        # GPU memory if available
        if torch.cuda.is_available():
            result["gpu_memory_alloc_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            result["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)

        return result

    def print_report(self, results: dict) -> None:
        print("\n── Inference Profile ──────────────────────────────")
        for k, v in results.items():
            print(f"  {k:<35} {v}")
        print("──────────────────────────────────────────────────\n")
