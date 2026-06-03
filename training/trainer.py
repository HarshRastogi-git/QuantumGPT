"""
QuantumGPT v2 — Training Engine
Supports both baseline and gated model training.
"""
import os
import math
import time
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Any

from model.transformer import QuantumGPT, GPTConfig


@dataclass
class TrainConfig:
    # I/O
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "model.pkl"
    log_dir: str = "logs"
    # Batch
    batch_size: int = 32
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # LR schedule (cosine with warmup)
    max_iters: int = 5000
    warmup_iters: int = 200
    min_lr: float = 1e-5
    # Eval
    eval_interval: int = 200
    eval_iters: int = 50
    # Device
    device: str = "cpu"
    # Mixed precision
    use_amp: bool = False
    # Gate optimizer (separate, fixed LR — not subject to cosine decay)
    gate_lr: float = 5e-2   # fixed LR for gates — not subject to cosine decay


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(cfg.warmup_iters, 1)
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(cfg.max_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


class Trainer:
    def __init__(
        self,
        model: QuantumGPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.model.to(self.device)

        self.optimizer, self.gate_optimizer = self._build_optimizers()
        self.scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

        self.iter_num = 0
        self.best_val_loss = float("inf")
        self.train_losses: list = []
        self.val_losses: list = []

        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(train_cfg.log_dir, exist_ok=True)

    def _build_optimizers(self):
        """
        Two optimizers:
          1. main_optimizer  — all non-gate params, AdamW with cosine LR decay
          2. gate_optimizer  — only g_raw params, Adam with FIXED lr=gate_lr

        Why separate? The cosine schedule decays the main LR to ~1e-5 by
        late training. Gate regularization only kicks in during phase 2
        (by default iter 3000+), so gate updates would be ~5e-8/step —
        completely invisible. A fixed gate_lr=1e-2 keeps gate learning
        strong throughout phase 2 regardless of where the main LR is.
        """
        gate_params = []
        other_decay = []
        other_nodecay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("g_raw"):
                gate_params.append(param)
            elif param.dim() >= 2:
                other_decay.append(param)
            else:
                other_nodecay.append(param)

        main_groups = [
            {"params": other_decay,   "weight_decay": self.cfg.weight_decay},
            {"params": other_nodecay, "weight_decay": 0.0},
        ]
        main_opt = torch.optim.AdamW(
            main_groups,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

        # Gate optimizer — only created if there are gate params
        if gate_params:
            gate_opt = torch.optim.Adam(
                gate_params,
                lr=self.cfg.gate_lr,
                betas=(0.9, 0.99),
            )
            n_gate = sum(p.numel() for p in gate_params)
            print(f"  [Trainer] Gate optimizer: {n_gate} gate params | fixed lr={self.cfg.gate_lr} (no decay)")
        else:
            gate_opt = None

        return main_opt, gate_opt

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        self.model.eval()
        results = {}
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            losses = []
            for i, (x, y) in enumerate(loader):
                if i >= self.cfg.eval_iters:
                    break
                x, y = x.to(self.device), y.to(self.device)
                # Pass iter info so gate reg is correctly gated during eval
                _, loss = self.model(x, y,
                                     current_iter=self.iter_num,
                                     max_iter=self.cfg.max_iters)
                losses.append(loss.item())
            results[split] = sum(losses) / max(len(losses), 1)
        self.model.train()
        return results

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  Training: {self.model}")
        print(f"  Device: {self.device}")
        print(f"  Iters: {self.cfg.max_iters} | Batch: {self.cfg.batch_size}")
        print(f"  Main LR: {self.cfg.learning_rate} → {self.cfg.min_lr}")
        if self.gate_optimizer is not None:
            print(f"  Gate LR: {self.cfg.gate_lr} (fixed, no decay)")
        print(f"{'='*60}\n")

        self.model.train()
        train_iter = iter(self.train_loader)
        t0 = time.time()
        tokens_processed = 0

        for self.iter_num in range(self.cfg.max_iters):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Update main LR (cosine decay)
            lr = get_lr(self.iter_num, self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                _, loss = self.model(x, y,
                                     current_iter=self.iter_num,
                                     max_iter=self.cfg.max_iters)

            # Backward
            self.scaler.scale(loss).backward()

            # Grad clipping
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            # Step both optimizers
            self.scaler.step(self.optimizer)
            if self.gate_optimizer is not None:
                self.gate_optimizer.step()

            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.gate_optimizer is not None:
                self.gate_optimizer.zero_grad(set_to_none=True)

            tokens_processed += x.numel()

            # Logging
            if self.iter_num % 50 == 0:
                elapsed = time.time() - t0
                tok_per_sec = tokens_processed / max(elapsed, 1e-6)
                self.train_losses.append((self.iter_num, loss.item()))
                print(
                    f"  iter {self.iter_num:5d}/{self.cfg.max_iters}"
                    f" | loss {loss.item():.4f}"
                    f" | lr {lr:.2e}"
                    f" | tok/s {tok_per_sec:.0f}"
                )

            # Evaluation + checkpoint
            if self.iter_num % self.cfg.eval_interval == 0 and self.iter_num > 0:
                losses = self.estimate_loss()
                val_loss = losses["val"]
                self.val_losses.append((self.iter_num, val_loss))
                val_ppl = math.exp(min(val_loss, 20))
                print(f"\n  ── Eval @ iter {self.iter_num} ──")
                print(f"  Train loss: {losses['train']:.4f} | Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                if self.model.config.use_gates:
                    gate_vals = self.model.get_gate_values()
                    threshold = self.model.config.gate_threshold
                    phase_start = int(self.model.config.gate_reg_start * self.cfg.max_iters)
                    phase = "Phase 1 — task only" if self.iter_num < phase_start else "Phase 2 — reg ON"
                    total = sum(len(v) for v in gate_vals.values())
                    below = sum(1 for v in gate_vals.values() for g in v if g < threshold)
                    above = sum(1 for v in gate_vals.values() for g in v if g > 0.5)
                    print(f"  [{phase}]  Gates below {threshold:.2f}: {below}/{total}  above 0.5: {above}/{total}")
                    for layer, gates in gate_vals.items():
                        gstr = " ".join(f"{g:.3f}" for g in gates)
                        flags = " ".join("↓" if g < threshold else ("▲" if g > 0.5 else "·") for g in gates)
                        print(f"    {layer}: [{gstr}]  {flags}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(tag="best")
                    print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
                print()

        self._save_checkpoint(tag=None)
        elapsed_total = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  Training complete in {elapsed_total/60:.1f} min")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoint: {self._ckpt_path()}")
        print(f"{'='*60}\n")

        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_iters": self.cfg.max_iters,
        }
        log_path = os.path.join(self.cfg.log_dir, self.cfg.checkpoint_name + ".json")
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        return history

    def _ckpt_path(self, tag: Optional[str] = None) -> str:
        name = self.cfg.checkpoint_name
        if tag:
            name = f"{tag}_{name}"
        return os.path.join(self.cfg.checkpoint_dir, name)

    def _save_checkpoint(self, tag: Optional[str] = None) -> None:
        path = self._ckpt_path(tag)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_config": self.model.config.__dict__,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)


def load_checkpoint(path: str, device: str = "cpu") -> QuantumGPT:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    config = GPTConfig(**ckpt["model_config"])
    model = QuantumGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model