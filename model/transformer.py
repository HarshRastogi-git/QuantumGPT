"""
QuantumGPT v2 — Core Transformer Architecture
Decoder-only GPT with Adaptive Gated Head Pruning (Structural Pruning).
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class GPTConfig:
    vocab_size: int = 4000
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1
    bias: bool = False
    # ── Gating ──────────────────────────────────────────────────────────────
    use_gates: bool = False
    gate_reg_lambda: float = 1e-2   
    gate_threshold: float = 0.2     
    gate_prune_pct: float = 0.33    
    gate_reg_start: float = 0.3     
    gate_temp_start: float = 4.0    
    gate_temp_end: float = 10.0     
    gate_binaryness: float = 0.5    

    @property
    def head_size(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config   = config
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_size = config.head_size

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

        if config.use_gates:
            self.g_raw = nn.Parameter(0.05 * torch.randn(config.n_head))
        else:
            self.g_raw = None

        self._current_temp: float = config.gate_temp_start
        self.active_heads: Optional[list] = None
        self.register_buffer("active_gates", None)

    def _gate_probs(self, temp: Optional[float] = None) -> torch.Tensor:
        t = temp if temp is not None else self._current_temp
        return torch.sigmoid(self.g_raw * t)

    def get_gates(self) -> torch.Tensor:
        if self.g_raw is None:
            return torch.ones(self.n_head)
        with torch.no_grad():
            return self._gate_probs()

    def set_temp(self, temp: float) -> None:
        self._current_temp = temp

    def gate_reg_loss(self) -> torch.Tensor:
        if self.g_raw is None:
            return torch.zeros(1).squeeze()
        g = self._gate_probs()
        return (g * (1.0 - g)).mean()

    def prune_heads(self, threshold: float, prune_pct: float) -> int:
        """Determines which heads to prune and freezes survival gates."""
        if self.g_raw is None:
            return 0

        gates = self.get_gates().tolist()
        n_prune_rank = max(0, round(self.n_head * prune_pct))
        ranked       = sorted(range(self.n_head), key=lambda i: gates[i])
        
        # Protect healthy heads (only rank-prune if < 0.5)
        prune_by_rank   = {i for i in ranked[:n_prune_rank] if gates[i] < 0.5}
        prune_by_thresh = {i for i, g in enumerate(gates) if g < threshold}
        pruned_set      = prune_by_rank | prune_by_thresh

        self.active_heads = [i for i in range(self.n_head) if i not in pruned_set]
        
        device = self.c_proj.weight.device
        active_gate_vals = [gates[i] for i in self.active_heads]
        self.register_buffer("active_gates", torch.tensor(active_gate_vals, dtype=torch.float32, device=device))

        self.g_raw = None
        return len(pruned_set)

    def structurally_prune(self):
        """
        Physically shrinks the weight matrices to permanently remove pruned heads.
        Fuses the gate values into the projection weights to eliminate overhead.
        """
        if self.active_heads is None or len(self.active_heads) == self.n_head:
            return 

        device = self.c_proj.weight.device
        n_active = len(self.active_heads)
        hs = self.head_size

        # 1. Shrink c_attn (Extract Q, K, V rows for active heads)
        # nn.Linear weight is (out_features, in_features)
        # keep_attn = []
        # for h in self.active_heads:
        #     keep_attn.extend(range(h * hs, (h + 1) * hs))                      # Q
        #     keep_attn.extend(range(self.n_embd + h * hs, self.n_embd + (h + 1) * hs))  # K
        #     keep_attn.extend(range(2 * self.n_embd + h * hs, 2 * self.n_embd + (h + 1) * hs)) # V
        
        # keep_attn_idx = torch.tensor(keep_attn, device=device)
        # new_attn_w = self.c_attn.weight.data[keep_attn_idx, :]
        # new_attn_b = self.c_attn.bias.data[keep_attn_idx] if self.config.bias else None

        # self.c_attn = nn.Linear(self.n_embd, 3 * n_active * hs, bias=self.config.bias).to(device)
        # self.c_attn.weight.data = new_attn_w
        # if self.config.bias: self.c_attn.bias.data = new_attn_b

        # 1. Shrink c_attn (Extract Q, K, V rows for active heads)
        keep_q, keep_k, keep_v = [], [], []
        for h in self.active_heads:
            keep_q.extend(range(h * hs, (h + 1) * hs))                      # Q
            keep_k.extend(range(self.n_embd + h * hs, self.n_embd + (h + 1) * hs))  # K
            keep_v.extend(range(2 * self.n_embd + h * hs, 2 * self.n_embd + (h + 1) * hs)) # V
        
        # Group them properly so .split() works in the forward pass!
        keep_attn_idx = torch.tensor(keep_q + keep_k + keep_v, device=device)
        
        new_attn_w = self.c_attn.weight.data[keep_attn_idx, :]
        new_attn_b = self.c_attn.bias.data[keep_attn_idx] if self.config.bias else None

        self.c_attn = nn.Linear(self.n_embd, 3 * n_active * hs, bias=self.config.bias).to(device)
        self.c_attn.weight.data = new_attn_w
        if self.config.bias: self.c_attn.bias.data = new_attn_b
        
        # 2. Shrink c_proj (Extract columns for active heads)
        keep_proj = []
        for h in self.active_heads:
            keep_proj.extend(range(h * hs, (h + 1) * hs))
        keep_proj_idx = torch.tensor(keep_proj, device=device)

        new_proj_w = self.c_proj.weight.data[:, keep_proj_idx]

        # FUSE GATES INTO WEIGHTS (Multiply columns by their respective gate value)
        for i, g in enumerate(self.active_gates):
            new_proj_w[:, i * hs : (i + 1) * hs] *= g

        new_proj_b = self.c_proj.bias.data if self.config.bias else None

        self.c_proj = nn.Linear(n_active * hs, self.n_embd, bias=self.config.bias).to(device)
        self.c_proj.weight.data = new_proj_w
        if self.config.bias: self.c_proj.bias.data = new_proj_b

        # 3. Update configuration for dense execution
        self.n_head = n_active
        self.active_heads = None
        self.active_gates = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        # Splits perfectly because we shrunk c_attn to exactly 3 * n_head * head_size
        q, k, v = qkv.split(self.n_head * self.head_size, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)

        # Pure Dense PyTorch Math (No overhead)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y   = att @ v                           

        # Used ONLY during training
        if self.g_raw is not None and self.training:
            gates = self._gate_probs(self._current_temp)
            y = y * gates.view(1, self.n_head, 1, 1)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class QuantumGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte":  nn.Embedding(config.vocab_size, config.n_embd),
            "wpe":  nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h":    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_gate_temp(self, current_iter: int, max_iter: int) -> float:
        cfg = self.config
        phase_start = int(cfg.gate_reg_start * max_iter)
        if current_iter < phase_start:
            return cfg.gate_temp_start
        progress = (current_iter - phase_start) / max(max_iter - phase_start, 1)
        return cfg.gate_temp_start + progress * (cfg.gate_temp_end - cfg.gate_temp_start)

    def forward(self, idx, targets=None, current_iter=0, max_iter=1):
        B, T = idx.shape
        assert T <= self.config.block_size

        if self.config.use_gates:
            current_temp = self._compute_gate_temp(current_iter, max_iter)
            for block in self.transformer["h"]:
                block.attn.set_temp(current_temp)

        pos     = torch.arange(T, device=idx.device)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x      = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if self.config.use_gates and self.config.gate_reg_lambda > 0:
                phase_start = int(self.config.gate_reg_start * max_iter)
                if current_iter >= phase_start:
                    gate_loss = torch.stack(
                        [blk.attn.gate_reg_loss() for blk in self.transformer["h"]]
                    ).sum() * self.config.gate_reg_lambda
                    loss = loss + gate_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95, greedy=False):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            t0      = time.perf_counter()
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :]

            if greedy:
                next_token = logits_last.argmax(dim=-1, keepdim=True)
            else:
                logits = logits_last / temperature

                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
                else:
                    next_token = torch.multinomial(probs, 1)

            idx = torch.cat([idx, next_token], dim=1)
            yield next_token.item(), (time.perf_counter() - t0) * 1000

    def prune_heads(self) -> dict:
        cfg   = self.config
        stats = {"total_heads": 0, "pruned_heads": 0, "active_heads": 0, "by_layer": []}
        for i, block in enumerate(self.transformer["h"]):
            total  = cfg.n_head
            pruned = block.attn.prune_heads(cfg.gate_threshold, cfg.gate_prune_pct)
            active = total - pruned
            stats["total_heads"]  += total
            stats["pruned_heads"] += pruned
            stats["active_heads"] += active
            stats["by_layer"].append({"layer": i, "active": active, "pruned": pruned})
        stats["prune_pct"] = 100.0 * stats["pruned_heads"] / max(stats["total_heads"], 1)
        return stats

    def structurally_prune(self):
        """Triggers physical structural compression across all layers."""
        for block in self.transformer["h"]:
            block.attn.structurally_prune()

    def get_gate_values(self) -> dict:
        result = {}
        for i, block in enumerate(self.transformer["h"]):
            if block.attn.g_raw is not None:
                result[f"layer_{i}"] = block.attn.get_gates().detach().cpu().tolist()
        return result

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def model_size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"QuantumGPT(layers={self.config.n_layer}, embd={self.config.n_embd}, "
            f"params={self.num_parameters():,}, "
            f"gates={'on' if self.config.use_gates else 'off'})"
        )

