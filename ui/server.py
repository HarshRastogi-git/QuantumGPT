"""
QuantumGPT v2 — Flask Web Server

Run from ANY directory:
    python ui/server.py
    cd ui && python server.py

All paths are resolved relative to this file's location.
"""
import json
import os
import pickle
import sys
import time
import threading
from typing import Generator, List, Optional

import torch
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# ── Resolve project root regardless of working directory ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../quantumgpt/ui/
_ROOT = os.path.dirname(_HERE)                     # .../quantumgpt/

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import GPTConfig, QuantumGPT

# ── Flask (absolute paths so templates work from any CWD) ────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(_HERE, "templates"),
    static_folder=os.path.join(_HERE, "static"),
)

# ── Checkpoint paths — match train.py / train_gated.py output names ───────────
TOKENIZER_PATH = os.path.join(_ROOT, "tokenizer", "tokenizer.json")
CHECKPOINTS = {
    "baseline": os.path.join(_ROOT, "checkpoints", "baseline.pkl"),
    "gated":    os.path.join(_ROOT, "checkpoints", "gated_v5.pkl"),
}
ABLATION_PATH = os.path.join(_ROOT, "checkpoints", "ablation_results.json")

# Back-fill gate config fields absent from old baseline checkpoints
_GATE_DEFAULTS = {
    "use_gates":       False,
    "gate_reg_lambda": 0.0,
    "gate_threshold":  0.2,
    "gate_prune_pct":  0.33,
    "gate_reg_start":  0.3,
    "gate_temp_start": 4.0,
    "gate_temp_end":   10.0,
    "gate_binaryness": 0.5,
}

_lock = threading.Lock()


def _gate_matrix(model: Optional[QuantumGPT]) -> List[List[float]]:
    if model is None:
        return []
    rows: List[List[float]] = []
    for blk in model.transformer["h"]:
        attn = blk.attn
        n = attn.n_head
        if attn.g_raw is not None:
            with torch.no_grad():
                gates = torch.sigmoid(attn.g_raw * attn._current_temp).tolist()
        elif attn.active_gates is not None:
            gates = attn.active_gates.detach().cpu().tolist()
        else:
            gates = [1.0] * n
        rows.append([round(float(g), 4) for g in gates])
    return rows


# ─────────────────────────────────────────────────────────────────────────── #
#  Model Manager                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class ModelManager:
    def __init__(self):
        self.model:         Optional[QuantumGPT]   = None
        self.tokenizer:     Optional[BPETokenizer] = None
        self.current_model: str  = "none"
        self.metrics:       dict = {}
        self.prune_stats:   dict = {}
        self.load_status:   str  = "idle"
        self.device = torch.device("cpu")

    def load_tokenizer(self):
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(
                f"Tokenizer not found: {TOKENIZER_PATH}\n"
                f"Run:  python train.py"
            )
        self.tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    def _build_model(self, path: str) -> QuantumGPT:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)

        cfg = ckpt["model_config"]
        for k, v in _GATE_DEFAULTS.items():
            cfg.setdefault(k, v)

        valid = set(GPTConfig.__dataclass_fields__)
        for stale in [k for k in list(cfg) if k not in valid]:
            cfg.pop(stale)

        config = GPTConfig(**cfg)
        model  = QuantumGPT(config)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)

        if config.use_gates:
            for block in model.transformer["h"]:
                block.attn.set_temp(config.gate_temp_end)
            self.prune_stats = model.prune_heads()
            model.structurally_prune()
            print(
                f"[server] Pruned {self.prune_stats['pruned_heads']} heads "
                f"({self.prune_stats['prune_pct']:.1f}%)"
            )
        else:
            self.prune_stats = {}

        model.eval()
        return model

    def load_model(self, model_name: str) -> dict:
        path = CHECKPOINTS.get(model_name)
        if not path:
            raise ValueError(f"Unknown model: {model_name!r}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Train first:  python train.py  /  python train_gated.py"
            )

        self.load_status = "loading"
        t0    = time.time()
        model = self._build_model(path)

        with _lock:
            self.model         = model
            self.current_model = model_name

        elapsed    = time.time() - t0
        cfg        = model.config
        orig_heads = cfg.n_layer * cfg.n_head
        live_heads = sum(b.attn.n_head for b in model.transformer["h"])

        self.metrics = {
            "model_name":   model_name,
            "parameters":   model.num_parameters(),
            "size_mb":      round(model.model_size_mb(), 2),
            "total_heads":  orig_heads,
            "active_heads": live_heads,
            "pruned_heads": orig_heads - live_heads,
            "prune_pct":    round(100.0 * (orig_heads - live_heads) / max(orig_heads, 1), 1),
            "n_layer":      cfg.n_layer,
            "n_head":       cfg.n_head,
            "n_embd":       cfg.n_embd,
            "vocab_size":   cfg.vocab_size,
            "use_gates":    cfg.use_gates,
            "load_time_ms": round(elapsed * 1000, 1),
            "by_layer":     self.prune_stats.get("by_layer", []),
        }
        self.load_status = "ready"
        return self.metrics

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        greedy: bool = False,
    ) -> Generator[str, None, None]:
        if self.model is None or self.tokenizer is None:
            yield f"data: {json.dumps({'error': 'No model loaded'})}\n\n"
            return

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        token_count   = 0
        total_latency = 0.0
        generated_ids: list = []
        prev_text = self.tokenizer.decode(prompt_ids)
        t_wall0 = time.perf_counter()

        for tok_id, elapsed_ms in self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        ):
            token_count   += 1
            total_latency += elapsed_ms
            generated_ids.append(tok_id)

            full_text = self.tokenizer.decode(prompt_ids + generated_ids)
            chunk     = full_text[len(prev_text):]
            prev_text = full_text

            avg_ms     = total_latency / token_count
            throughput = 1000.0 / max(avg_ms, 1e-6)

            yield (
                f"data: {json.dumps({'token': chunk, 'token_count': token_count, 'latency_ms': round(avg_ms, 3), 'throughput': round(throughput, 1), 'done': False})}\n\n"
            )

        wall_elapsed = time.perf_counter() - t_wall0
        avg_ms       = total_latency / max(token_count, 1)
        final_tps    = token_count / max(wall_elapsed, 1e-9)
        full_out     = self.tokenizer.decode(prompt_ids + generated_ids)
        yield f"data: {json.dumps({'done': True, 'full_text': full_out, 'total_tokens': token_count, 'avg_latency_ms': round(avg_ms, 3), 'elapsed_s': round(wall_elapsed, 2), 'tps': round(final_tps, 1)})}\n\n"


manager = ModelManager()


def _saved_ppl_for_label(label: str) -> Optional[float]:
    if not os.path.exists(ABLATION_PATH):
        return None
    try:
        with open(ABLATION_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        key = "gated" if label == "gated" else "baseline"
        v = saved.get(key, {}).get("perplexity")
        return float(v) if v is not None else None
    except Exception:
        return None


def _stats_payload() -> dict:
    """Shape expected by ui/static/js/app.js fetchStatus()."""
    m = manager.model
    tok = manager.tokenizer
    label = manager.current_model

    if m is None or tok is None or label == "none":
        return {
            "ready": False,
            "label": None,
            "available": {k: os.path.exists(v) for k, v in CHECKPOINTS.items()},
        }

    met = manager.metrics
    cfg = m.config
    total_heads = met["total_heads"]
    live_heads  = met["active_heads"]

    return {
        "ready":        True,
        "label":        label,
        "params":       met["parameters"],
        "size_mb":      met["size_mb"],
        "vocab_size":   len(tok) if tok else met.get("vocab_size", 0),
        "block_size":   cfg.block_size,
        "n_layer":      cfg.n_layer,
        "n_head":       cfg.n_head,
        "n_embd":       cfg.n_embd,
        "live_heads":   live_heads,
        "total_heads":  total_heads,
        "dead_heads":   total_heads - live_heads,
        "sparsity_pct": round((total_heads - live_heads) / max(total_heads, 1) * 100, 1),
        "use_gates":    cfg.use_gates,
        "gate_matrix":  _gate_matrix(m),
        "saved_ppl":    _saved_ppl_for_label(label),
        "available":    {k: os.path.exists(v) for k, v in CHECKPOINTS.items()},
        "status":       manager.load_status,
        "current_model": label,
    }


# ─────────────────────────────────────────────────────────────────────────── #
#  Routes                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

@app.route("/")
def index():
    return render_template("index.html")


def _load_model_handler():
    data       = request.get_json() or {}
    model_name = data.get("model") or data.get("label", "baseline")
    try:
        if manager.tokenizer is None:
            manager.load_tokenizer()
        metrics = manager.load_model(model_name)
        return jsonify({"success": True, "message": f"Loaded {model_name}", "metrics": metrics, "label": manager.current_model})
    except FileNotFoundError as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 404
    except Exception as e:
        manager.load_status = "error"
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    return _load_model_handler()


@app.route("/api/load", methods=["POST"])
def api_load():
    """Alias for app.js — POST JSON { \"label\": \"baseline\" | \"gated\" }."""
    return _load_model_handler()


@app.route("/api/generate", methods=["POST"])
def api_generate():
    if manager.model is None:
        return jsonify({"error": "No model loaded — POST to /api/load first."}), 400

    data        = request.get_json() or {}
    prompt      = data.get("prompt", "")
    max_tokens  = min(int(data.get("max_tokens", 200)), 600)
    temperature = float(data.get("temperature", 0.8))
    top_k       = int(data.get("top_k", 50))
    top_p       = float(data.get("top_p", 0.95))
    greedy      = bool(data.get("greedy", False))

    def stream():
        yield from manager.generate_stream(
            prompt, max_tokens, temperature, top_k, top_p, greedy
        )

    return Response(
        stream_with_context(stream()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    if not manager.metrics:
        return jsonify({"error": "No model loaded"}), 404
    return jsonify(manager.metrics)


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(_stats_payload())


@app.route("/api/gate_report")
def api_gate_report():
    m = manager.model
    if m is None:
        return jsonify({"error": "No model"}), 503

    cfg = m.config
    report = []
    for i, blk in enumerate(m.transformer["h"]):
        attn = blk.attn
        if attn.g_raw is not None:
            with torch.no_grad():
                gates = torch.sigmoid(attn.g_raw * attn._current_temp).tolist()
        else:
            gates = [1.0] * attn.n_head

        gates_sorted = sorted(gates, reverse=True)
        dead_sorted  = [g < cfg.gate_threshold for g in gates_sorted]

        report.append({
            "layer":      i,
            "n_head":     attn.n_head,
            "gates":      [round(g, 4) for g in gates_sorted],
            "dead":       dead_sorted,
            "live_count": sum(1 for d in dead_sorted if not d),
        })

    return jsonify({
        "report":      report,
        "live_heads":  sum(r["live_count"] for r in report),
        "total_heads": sum(r["n_head"] for r in report),
    })


@app.route("/api/ablation")
def api_ablation():
    if not os.path.exists(ABLATION_PATH):
        return jsonify({"error": "Run evaluate.py first"}), 404
    with open(ABLATION_PATH, encoding="utf-8") as f:
        return jsonify(json.load(f))


# ─────────────────────────────────────────────────────────────────────────── #
#  Startup                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def startup():
    try:
        manager.load_tokenizer()
        print("[server] Tokenizer loaded.")
    except Exception as e:
        print(f"[server] WARNING — tokenizer: {e}")
        return

    # Prefer gated checkpoint if present, else baseline (both remain loadable via /api/load).
    for name in ("gated", "baseline"):
        path = CHECKPOINTS.get(name)
        if path and os.path.exists(path):
            print(f"[server] Auto-loading '{name}' from {os.path.basename(path)}...")
            try:
                manager.load_model(name)
                m = manager.metrics
                print(
                    f"[server] ✓ Ready — {m['parameters']:,} params | "
                    f"{m['active_heads']}/{m['total_heads']} heads active"
                )
            except Exception as e:
                print(f"[server] Failed to load {name}: {e}")
            break
    else:
        print("[server] No checkpoints found.")
        print("[server] Train first:  python train.py && python train_gated.py")


if __name__ == "__main__":
    print("\n" + "═" * 52)
    print("  QuantumGPT v2  ·  UI Server")
    print("  http://localhost:5000")
    print("═" * 52 + "\n")
    startup()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


# """
# QuantumGPT v2 — UI Server
# =========================
# Run from your project root:
#     pip install flask
#     python ui/server.py

# The gated model is structurally pruned immediately after loading —
# exactly as evaluate.py does — so generation is always coherent.
# """

# import json
# import os
# import pickle
# import sys
# import time

# import torch
# import torch.nn.functional as F
# from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from model.transformer import GPTConfig, QuantumGPT
# from tokenizer.bpe_tokenizer import BPETokenizer

# # ── Paths ──────────────────────────────────────────────────────────────────

# ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CKPT_DIR = os.path.join(ROOT, "checkpoints")
# TOK_PATH = os.path.join(ROOT, "tokenizer", "tokenizer.json")

# CHECKPOINTS = {
#     "gated":    "gated_v5.pkl",
#     "baseline": "baseline.pkl",
# }

# _GATE_DEFAULTS = {
#     "use_gates": False, "gate_reg_lambda": 0.0, "gate_threshold": 0.2,
#     "gate_prune_pct": 0.33, "gate_reg_start": 0.3,
#     "gate_temp_start": 4.0, "gate_temp_end": 10.0, "gate_binaryness": 0.5,
# }

# app = Flask(__name__, template_folder="templates", static_folder="static")

# # ── Global state ───────────────────────────────────────────────────────────

# _state = {
#     "model":      None,
#     "tokenizer":  None,
#     "label":      None,
#     "loading":    False,
#     "prune_stats": None,   # stored for the gate panel after pruning
# }


# # ── Model loading ──────────────────────────────────────────────────────────

# def _load(label: str):
#     fname = CHECKPOINTS.get(label)
#     if not fname:
#         return False, f"Unknown checkpoint: {label}"

#     path = os.path.join(CKPT_DIR, fname)
#     if not os.path.exists(path):
#         return False, f"File not found: {path}"

#     try:
#         _state["loading"] = True

#         with open(path, "rb") as f:
#             ckpt = pickle.load(f)

#         cfg_dict = ckpt["model_config"]
#         for k, v in _GATE_DEFAULTS.items():
#             cfg_dict.setdefault(k, v)
#         valid = set(GPTConfig.__dataclass_fields__)
#         for stale in [k for k in list(cfg_dict) if k not in valid]:
#             cfg_dict.pop(stale)

#         config = GPTConfig(**cfg_dict)
#         model  = QuantumGPT(config)
#         model.load_state_dict(ckpt["model_state"])

#         prune_stats = None

#         # ── KEY FIX ────────────────────────────────────────────────────────
#         # Gated checkpoints have live gate parameters (g_raw != None).
#         # At inference self.training=False so the gate condition
#         #   `if self.g_raw is not None and self.training`
#         # never fires — meaning heads that were trained with gates near 0
#         # now run at full strength, causing a 10× signal mismatch that
#         # produces incoherent output.
#         #
#         # Calling prune_heads() sets g_raw=None and records active_heads.
#         # Calling structurally_prune() physically removes the dead-head
#         # rows/columns and fuses surviving gate values into the projection
#         # weights. After this the model is internally consistent and
#         # generates coherently — this is identical to what evaluate.py does.
#         # ──────────────────────────────────────────────────────────────────
#         if config.use_gates:
#             # Use final temperature so gate decisions match training intent
#             final_temp = config.gate_temp_end
#             for block in model.transformer["h"]:
#                 block.attn.set_temp(final_temp)

#             prune_stats = model.prune_heads()
#             model.structurally_prune()
#             print(f"[Load] Structural pruning applied: "
#                   f"{prune_stats['pruned_heads']} heads removed "
#                   f"({prune_stats['prune_pct']:.1f}%)")

#         model.eval()

#         tok = BPETokenizer.load(TOK_PATH)
#         _state["model"]       = model
#         _state["tokenizer"]   = tok
#         _state["label"]       = label
#         _state["prune_stats"] = prune_stats
#         _state["loading"]     = False
#         return True, f"Loaded {label}"

#     except Exception as e:
#         _state["loading"] = False
#         return False, str(e)


# # Boot
# for _boot in ["gated", "baseline"]:
#     _ok, _msg = _load(_boot)
#     if _ok:
#         print(f"[Boot] {_msg}")
#         break
# else:
#     print("[Boot] No checkpoints found — add .pkl files to checkpoints/")


# # ── Generation ─────────────────────────────────────────────────────────────

# def _generate_tokens(model, prompt_ids, max_tokens,
#                      temperature, top_k, top_p, greedy):
#     """
#     Clean autoregressive decode loop.

#     Does NOT rely on the training/eval mode distinction — after structural
#     pruning g_raw is None so gate logic is dead, and this function always
#     calls model() in a no_grad context with the model already in eval mode.
#     """
#     idx = torch.tensor([prompt_ids], dtype=torch.long)

#     with torch.no_grad():
#         for _ in range(max_tokens):
#             cond = (idx if idx.size(1) <= model.config.block_size
#                     else idx[:, -model.config.block_size:])

#             logits, _ = model(cond)
#             logits     = logits[:, -1, :]          # (1, vocab_size)

#             if greedy:
#                 nxt = int(logits.argmax(-1).item())
#             else:
#                 logits = logits / temperature

#                 if top_k > 0:
#                     thresh = torch.topk(
#                         logits, min(top_k, logits.size(-1))
#                     ).values[:, [-1]]
#                     logits = logits.masked_fill(logits < thresh, float("-inf"))

#                 probs = F.softmax(logits, dim=-1)

#                 if top_p < 1.0:
#                     sp, si = torch.sort(probs, descending=True)
#                     sp[torch.cumsum(sp, -1) - sp > top_p] = 0.0
#                     sp = sp / sp.sum(-1, keepdim=True)
#                     nxt = int(si.gather(-1, torch.multinomial(sp, 1)).item())
#                 else:
#                     nxt = int(torch.multinomial(probs, 1).item())

#             idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
#             yield nxt


# # ── Helpers ────────────────────────────────────────────────────────────────

# def _gate_matrix():
#     """
#     Returns a 2-D list [layer][head] of gate values for the heatmap.

#     After structural pruning: g_raw=None, active_gates=None.
#     We show the remaining n_active heads all at 1.0 (they are fully active —
#     their gate values were fused into the projection weights).
#     """
#     m = _state["model"]
#     if m is None:
#         return []
#     rows = []
#     for blk in m.transformer["h"]:
#         attn = blk.attn
#         if attn.g_raw is not None:
#             # Still has live gates (baseline model with use_gates=False never
#             # reaches here; this path is a safety net only)
#             with torch.no_grad():
#                 gates = torch.sigmoid(attn.g_raw * attn._current_temp).tolist()
#         else:
#             # Post-pruning: all surviving heads are fully active
#             gates = [1.0] * attn.n_head
#         rows.append([round(g, 4) for g in gates])
#     return rows


# def _stats():
#     m   = _state["model"]
#     tok = _state["tokenizer"]
#     if m is None:
#         return {"ready": False}

#     cfg = m.config

#     # n_head in config reflects the original count.
#     # After structural pruning, each block's attn.n_head holds the actual count.
#     total_heads = cfg.n_layer * cfg.n_head
#     live_heads  = sum(blk.attn.n_head for blk in m.transformer["h"])

#     size_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6
#     params  = sum(p.numel() for p in m.parameters())

#     T, C, L, V = cfg.block_size, cfg.n_embd, cfg.n_layer, cfg.vocab_size
#     flops = (L * (2*T*C*3*C + 4*T*T*C + 2*T*C*C + 2*T*C*4*C + 2*T*4*C*C)
#              + 2*T*C*V)

#     # Pull saved perplexity from benchmarks/ablation_results.json
#     ppl = None
#     mp  = os.path.join(ROOT, "benchmarks", "ablation_results.json")
#     if os.path.exists(mp):
#         try:
#             with open(mp) as f:
#                 saved = json.load(f)
#             key = "gated" if _state["label"] == "gated" else "baseline"
#             entry = saved.get(key) or {}
#             ppl = entry.get("perplexity")
#         except Exception:
#             pass

#     return {
#         "ready":        True,
#         "label":        _state["label"],
#         "params":       params,
#         "size_mb":      round(size_mb, 3),
#         "vocab_size":   len(tok) if tok else 0,
#         "block_size":   cfg.block_size,
#         "n_layer":      cfg.n_layer,
#         "n_head":       cfg.n_head,
#         "n_embd":       cfg.n_embd,
#         "live_heads":   live_heads,
#         "total_heads":  total_heads,
#         "dead_heads":   total_heads - live_heads,
#         "sparsity_pct": round((total_heads - live_heads) / max(total_heads, 1) * 100, 1),
#         "use_gates":    cfg.use_gates,
#         "gate_matrix":  _gate_matrix(),
#         "flops":        flops,
#         "saved_ppl":    ppl,
#         "available":    {k: os.path.exists(os.path.join(CKPT_DIR, v))
#                          for k, v in CHECKPOINTS.items()},
#     }


# # ── Routes ─────────────────────────────────────────────────────────────────

# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/api/status")
# def status():
#     return jsonify(_stats())


# @app.route("/api/load", methods=["POST"])
# def load():
#     label = request.get_json(force=True).get("label", "gated")
#     ok, msg = _load(label)
#     return jsonify({"success": ok, "message": msg, "label": _state["label"]})


# @app.route("/api/generate", methods=["POST"])
# def gen():
#     data        = request.get_json(force=True)
#     prompt      = data.get("prompt", "")
#     max_tokens  = min(int(data.get("max_tokens", 300)), 600)
#     temperature = float(data.get("temperature", 0.8))
#     top_k       = int(data.get("top_k", 50))
#     top_p       = float(data.get("top_p", 0.95))
#     greedy      = bool(data.get("greedy", False))

#     model = _state["model"]
#     tok   = _state["tokenizer"]
#     if model is None or tok is None:
#         return jsonify({"error": "Model not loaded"}), 503

#     def stream():
#         ids = tok.encode(prompt)
#         if not ids:
#             yield f"data: {json.dumps({'error': 'Empty prompt'})}\n\n"
#             return

#         generated_ids = []
#         t0 = time.perf_counter()

#         for i, nxt in enumerate(_generate_tokens(
#             model, ids, max_tokens, temperature, top_k, top_p, greedy
#         )):
#             generated_ids.append(nxt)
#             tok_str = tok.decode([nxt])
#             yield f"data: {json.dumps({'token': tok_str, 'index': i, 'done': False})}\n\n"

#         elapsed  = time.perf_counter() - t0
#         full_txt = tok.decode(ids + generated_ids)
#         yield f"data: {json.dumps({'done': True, 'full_text': full_txt, 'elapsed_s': round(elapsed, 2), 'tps': round(max_tokens / elapsed, 1)})}\n\n"

#     # Fix: set implicit_sequence_conversion=False so Werkzeug does NOT call
#     # list(generator) to check Content-Length — that would run the entire
#     # generator before sending the first byte, defeating streaming entirely.
#     resp = Response(
#         stream_with_context(stream()),
#         mimetype="text/event-stream",
#         headers={
#             "Cache-Control":    "no-cache",
#             "X-Accel-Buffering":"no",
#             "Connection":       "keep-alive",
#             "Transfer-Encoding":"chunked",
#         },
#     )
#     resp.implicit_sequence_conversion = False
#     return resp


# @app.route("/api/gate_report")
# def gate_report():
#     m  = _state["model"]
#     ps = _state["prune_stats"]
#     if m is None:
#         return jsonify({"error": "No model loaded"}), 503

#     report = []
#     for i, blk in enumerate(m.transformer["h"]):
#         attn = blk.attn

#         # Post-structural-pruning: g_raw is None, n_head holds active count.
#         # Show each surviving head at gate=1.0.
#         # If prune_stats are available, show the original gate values too.
#         orig_gates = None
#         if ps and i < len(ps.get("by_layer", [])):
#             # by_layer stores active/pruned counts from prune_heads()
#             pass  # gate values were consumed by prune_heads()

#         if attn.g_raw is not None:
#             with torch.no_grad():
#                 gates = torch.sigmoid(attn.g_raw * attn._current_temp).tolist()
#             dead = [g < m.config.gate_threshold for g in gates]
#         else:
#             gates = [1.0] * attn.n_head
#             dead  = [False] * attn.n_head

#         pruned_count = 0
#         if ps:
#             layer_info = next(
#                 (row for row in ps.get("by_layer", []) if row["layer"] == i), None
#             )
#             if layer_info:
#                 pruned_count = layer_info["pruned"]

#         report.append({
#             "layer":        i,
#             "n_head":       attn.n_head,          # live count after pruning
#             "n_pruned":     pruned_count,
#             "gates":        [round(g, 4) for g in gates],
#             "dead":         dead,
#             "live_count":   attn.n_head,
#         })

#     return jsonify({
#         "report":      report,
#         "live_heads":  sum(r["live_count"] for r in report),
#         "total_heads": sum(r["live_count"] + r["n_pruned"] for r in report),
#     })


# @app.route("/api/ablation")
# def ablation():
#     path = os.path.join(ROOT, "benchmarks", "ablation_results.json")
#     if not os.path.exists(path):
#         return jsonify({"error": "Run evaluate.py first"}), 404
#     with open(path) as f:
#         return jsonify(json.load(f))


# if __name__ == "__main__":
#     print("\n" + "═" * 52)
#     print("  QuantumGPT v2  ·  UI Server")
#     print("  http://localhost:5000")
#     print("═" * 52 + "\n")
#     app.run(debug=False, port=5000, threaded=True)