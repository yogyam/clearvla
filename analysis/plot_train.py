"""Plot train/val loss from results/week3/train_log_<run>.jsonl -> results/week3/train_curve_<run>.png"""
import sys, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else "full"
rows = [json.loads(l) for l in (ROOT / f"results/week3/train_log_{run}.jsonl").read_text().splitlines()]
tr = [(r["step"], r["loss"]) for r in rows if "loss" in r]; va = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
fig, ax = plt.subplots(figsize=(7, 3.6), dpi=130)
if tr: ax.plot(*zip(*tr), lw=1, label="train (100-step mean)")
if va: ax.plot(*zip(*va), "o-", ms=3, lw=1, label="val (EMA weights)")
ax.set_xlabel("step"); ax.set_ylabel("flow-matching MSE"); ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(); ax.set_title(f"Model A — {run}")
fig.tight_layout(); out = ROOT / f"results/week3/train_curve_{run}.png"; fig.savefig(out); print("wrote", out, "| last train", tr[-1] if tr else None, "| best val", min(va, key=lambda x: x[1]) if va else None)
