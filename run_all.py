"""
run_all.py — Full QuantumGPT v2 pipeline in one command.

Steps:
  1. Download corpus (Shakespeare + Gutenberg)
  2. Train tokenizer
  3. Train baseline model
  4. Train gated model
  5. Run ablation evaluation
  6. Launch UI

Usage:
    python run_all.py                    # full pipeline
    python run_all.py --skip_data        # skip re-download
    python run_all.py --iters 2000       # faster training
    python run_all.py --ui_only          # just launch UI
"""
import argparse
import os
import subprocess
import sys


def run(cmd: list, desc: str = "") -> int:
    if desc:
        print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5000, help="Training iterations per model")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip_data", action="store_true", help="Skip corpus download")
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_gated", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--ui_only", action="store_true", help="Skip training, launch UI")
    p.add_argument("--warmstart", action="store_true", help="Init gated from baseline weights")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("""
╔══════════════════════════════════════════════════════════╗
║           QuantumGPT v2 — Full Pipeline Runner           ║
║       Adaptive Gated Head Pruning · Research Demo        ║
╚══════════════════════════════════════════════════════════╝
    """)

    if args.ui_only:
        print("[run_all] Launching UI only...")
        os.execv(sys.executable, [sys.executable, os.path.join(base_dir, "ui/server.py")])
        return

    # 1. Data
    if not args.skip_data:
        if not os.path.exists(os.path.join(base_dir, "data/raw.txt")):
            rc = run(["data/prepare_data.py"], "Step 1: Downloading corpus")
            if rc != 0:
                print("WARNING: Data download failed. Creating minimal fallback...")
                os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
                fallback = "The quick brown fox. " * 20000
                with open(os.path.join(base_dir, "data/raw.txt"), "w") as f:
                    f.write(fallback)
        else:
            print("[run_all] Corpus already exists. Skipping download.")
    else:
        print("[run_all] Skipping data download (--skip_data).")

    # 2. Baseline training
    if not args.skip_baseline:
        cmd = ["train.py", "--iters", str(args.iters), "--batch", str(args.batch), "--device", args.device]
        rc = run(cmd, "Step 2: Training Baseline Model")
        if rc != 0:
            print("ERROR: Baseline training failed."); sys.exit(1)
    else:
        print("[run_all] Skipping baseline training (--skip_baseline).")

    # 3. Gated training
    if not args.skip_gated:
        cmd = ["train_gated.py", "--iters", str(args.iters), "--batch", str(args.batch), "--device", args.device]
        if args.warmstart:
            cmd.append("--warmstart")
        rc = run(cmd, "Step 3: Training Gated Model (Adaptive Head Pruning)")
        if rc != 0:
            print("ERROR: Gated training failed."); sys.exit(1)
    else:
        print("[run_all] Skipping gated training (--skip_gated).")

    # 4. Evaluation
    if not args.skip_eval:
        rc = run(["evaluate.py", "--device", args.device], "Step 4: Ablation Study")
        if rc != 0:
            print("WARNING: Evaluation had errors (non-fatal).")
    else:
        print("[run_all] Skipping evaluation (--skip_eval).")

    # 5. Launch UI
    print("\n" + "="*60)
    print("  All steps complete! Launching QuantumGPT v2 UI...")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")

    # Import and run server directly
    sys.path.insert(0, base_dir)
    from ui.server import app, startup
    startup()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
