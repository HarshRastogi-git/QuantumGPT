"""
smoke_test.py — Fast sanity check for all QuantumGPT v2 components.
Runs in under 60 seconds on CPU. No data download required.

Usage:
    python smoke_test.py
"""
import sys
import os
import time
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def test_tokenizer():
    section("1. BPE Tokenizer")
    from tokenizer.bpe_tokenizer import BPETokenizer

    sample = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles.
    It was the best of times, it was the worst of times.
    The quick brown fox jumps over the lazy dog.
    """ * 50

    tok = BPETokenizer(vocab_size=500)
    tok.train(sample, verbose=False)

    test_str = "To be or not to be"
    ids = tok.encode(test_str)
    decoded = tok.decode(ids)
    print(f"  Input:   '{test_str}'")
    print(f"  IDs:     {ids[:10]}...")
    print(f"  Decoded: '{decoded}'")
    print(f"  Vocab size: {len(tok)}")
    assert len(ids) > 0, "Encoding failed"
    assert len(tok) <= 500, "Vocab too large"
    print("  ✓ Tokenizer OK")

    # Save/load
    os.makedirs("smoke_out", exist_ok=True)
    tok.save("smoke_out/test_tok.json")
    tok2 = BPETokenizer.load("smoke_out/test_tok.json")
    ids2 = tok2.encode(test_str)
    assert ids == ids2, "Save/load mismatch"
    print("  ✓ Save/Load OK")
    return tok


def test_model(tokenizer):
    section("2. QuantumGPT Model (Baseline)")
    from model.transformer import QuantumGPT, GPTConfig

    cfg = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        use_gates=False,
    )
    model = QuantumGPT(cfg)
    print(f"  {model}")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Size: {model.model_size_mb():.3f} MB")

    x = torch.randint(0, len(tokenizer), (2, 32))
    y = torch.randint(0, len(tokenizer), (2, 32))
    logits, loss = model(x, y)
    print(f"  Forward: logits={logits.shape} loss={loss.item():.4f}")
    assert logits.shape == (2, 32, len(tokenizer))
    assert not math.isnan(loss.item())
    print("  ✓ Baseline model OK")
    return model, cfg


def test_gated_model(tokenizer):
    section("3. QuantumGPT Model (Gated)")
    from model.transformer import QuantumGPT, GPTConfig

    cfg = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        use_gates=True,
        gate_reg_lambda=0.01,
        gate_threshold=0.1,
    )
    model = QuantumGPT(cfg)

    x = torch.randint(0, len(tokenizer), (2, 32))
    y = torch.randint(0, len(tokenizer), (2, 32))
    logits, loss = model(x, y)
    print(f"  Forward: logits={logits.shape} loss={loss.item():.4f}")

    gate_vals = model.get_gate_values()
    print(f"  Gate values: {gate_vals}")

    loss.backward()
    # Check gates have gradients
    for block in model.transformer["h"]:
        if block.attn.head_gates_raw is not None:
            assert block.attn.head_gates_raw.grad is not None, "Gates have no gradient!"

    prune_stats = model.prune_heads()
    print(f"  Prune stats: {prune_stats}")
    print("  ✓ Gated model OK")


def test_generation(model, tokenizer):
    section("4. Text Generation")
    prompt = "To be or not"
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    tokens_generated = []
    t0 = time.time()
    for tok_id, ms in model.generate(idx, max_new_tokens=20, temperature=0.8):
        tokens_generated.append(tok_id)
    elapsed = time.time() - t0

    generated_text = tokenizer.decode(ids + tokens_generated)
    print(f"  Prompt: '{prompt}'")
    print(f"  Generated: '{generated_text}'")
    print(f"  Tokens: {len(tokens_generated)} in {elapsed:.2f}s")
    assert len(tokens_generated) == 20
    print("  ✓ Generation OK")


def test_training_step(tokenizer):
    section("5. Training Step")
    from model.transformer import QuantumGPT, GPTConfig

    cfg = GPTConfig(
        vocab_size=len(tokenizer), block_size=32, n_layer=2, n_head=2,
        n_embd=32, dropout=0.0, use_gates=True, gate_reg_lambda=1e-3,
    )
    model = QuantumGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(20):
        x = torch.randint(0, len(tokenizer), (4, 32))
        y = torch.randint(0, len(tokenizer), (4, 32))
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    # Training should generally reduce loss (not guaranteed in 20 steps with random data, but usually true)
    print("  ✓ Training step OK")


def test_checkpoint(tokenizer):
    section("6. Checkpoint Save/Load")
    import pickle
    from model.transformer import QuantumGPT, GPTConfig
    from training.trainer import TrainConfig

    cfg = GPTConfig(
        vocab_size=len(tokenizer), block_size=16, n_layer=2, n_head=2,
        n_embd=32, dropout=0.0, use_gates=False,
    )
    model = QuantumGPT(cfg)

    ckpt = {
        "model_state": model.state_dict(),
        "model_config": cfg.__dict__,
        "iter_num": 0,
        "best_val_loss": 9.99,
    }
    os.makedirs("smoke_out", exist_ok=True)
    path = "smoke_out/test_ckpt.pkl"
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)

    with open(path, "rb") as f:
        ckpt2 = pickle.load(f)
    cfg2 = GPTConfig(**ckpt2["model_config"])
    model2 = QuantumGPT(cfg2)
    model2.load_state_dict(ckpt2["model_state"])
    print(f"  Saved and reloaded: {model2}")
    print("  ✓ Checkpoint OK")


def cleanup():
    import shutil
    if os.path.exists("smoke_out"):
        shutil.rmtree("smoke_out")


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║         QuantumGPT v2 — Smoke Test Suite             ║
╚══════════════════════════════════════════════════════╝
    """)
    t0 = time.time()
    passed = 0
    failed = 0

    tests = [
        ("Tokenizer", lambda: test_tokenizer()),
    ]

    tok = None
    try:
        tok = test_tokenizer(); passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}"); failed += 1

    if tok:
        try:
            model, cfg = test_model(tok); passed += 1
            try:
                test_generation(model, tok); passed += 1
            except Exception as e:
                print(f"  ✗ Generation FAILED: {e}"); failed += 1
        except Exception as e:
            print(f"  ✗ Model FAILED: {e}"); failed += 1

        try:
            test_gated_model(tok); passed += 1
        except Exception as e:
            print(f"  ✗ Gated model FAILED: {e}"); failed += 1

        try:
            test_training_step(tok); passed += 1
        except Exception as e:
            print(f"  ✗ Training FAILED: {e}"); failed += 1

        try:
            test_checkpoint(tok); passed += 1
        except Exception as e:
            print(f"  ✗ Checkpoint FAILED: {e}"); failed += 1

    elapsed = time.time() - t0
    cleanup()

    print(f"\n{'═'*50}")
    print(f"  Results: {passed} passed, {failed} failed  ({elapsed:.1f}s)")
    if failed == 0:
        print("  ✓ All systems go — ready to train!")
    else:
        print("  ✗ Fix failures before training.")
    print(f"{'═'*50}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
