"""Validity-gate report: success per cell with 95% bootstrap CIs over paired seeds, glass-opaque gap per task,
failure histogram. Works for any config prefix (reused in Week 5).

  python analysis/gate.py --out results/week3 --config full
"""
import argparse, json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
TASKS = ("grasp", "pour", "insert"); MATERIALS = ("opaque", "glass")


def load(out, config, material):
    p = out / f"{config}_{material}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def boot_ci(x, n=10000, rng=None):
    x = np.asarray(x, float); rng = rng or np.random.default_rng(0)
    if len(x) == 0: return (float("nan"), float("nan"))
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1); return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def paired_gap_ci(a, b, n=10000):
    """a, b: dicts seed -> success. Returns mean(a-b) over shared seeds and its bootstrap CI."""
    seeds = sorted(set(a) & set(b)); d = np.array([float(a[s]) - float(b[s]) for s in seeds])
    if len(d) == 0: return float("nan"), (float("nan"), float("nan")), 0
    rng = np.random.default_rng(0); m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(d.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))), len(seeds)


def report(out, config, gate=(0.60, 0.15)):
    rows = {m: load(out, config, m) for m in MATERIALS}
    lines = [f"# Validity gate — config `{config}`\n", "| Task | Material | Success | 95% CI | n | Mean steps | Failures |", "|---|---|---|---|---|---|---|"]
    cells = {}; ok = True
    for t in TASKS:
        for m in MATERIALS:
            r = [x for x in rows[m] if x["task"] == t]; s = [x["success"] for x in r]
            rate = float(np.mean(s)) if s else float("nan"); lo, hi = boot_ci(s); cells[(t, m)] = {x["seed"]: x["success"] for x in r}
            fails = collections.Counter((x["failure"].split("_")[0] or "?") for x in r if not x["success"])
            lines.append(f"| {t} | {m} | **{100*rate:.0f}%** | [{100*lo:.0f}, {100*hi:.0f}] | {len(s)} | {np.mean([x['steps'] for x in r]) if r else float('nan'):.0f} | {', '.join(f'{k}×{v}' for k, v in fails.most_common(4)) or '—'} |")
            ok &= (rate >= gate[0]) if s else False
    lines.append("\n| Task | opaque − glass (pts) | 95% CI | paired n |"); lines.append("|---|---|---|---|")
    for t in TASKS:
        g, ci, n = paired_gap_ci(cells[(t, "opaque")], cells[(t, "glass")])
        lines.append(f"| {t} | {100*g:+.0f} | [{100*ci[0]:+.0f}, {100*ci[1]:+.0f}] | {n} |"); ok &= (abs(g) <= gate[1]) if n else False
    tim = [x for m in MATERIALS for x in rows[m]]
    if tim: lines.append(f"\nPer-observation timing: encoder {1000*np.mean([x['t_encoder'] for x in tim]):.0f} ms, policy {1000*np.mean([x['t_policy'] for x in tim]):.0f} ms; {np.mean([x['n_obs'] for x in tim]):.1f} observations/episode; {np.mean([x['wall'] for x in tim]):.1f} s/episode wall.")
    lines.append(f"\n**Gate (≥{100*gate[0]:.0f}% every cell, |gap| ≤ {100*gate[1]:.0f} pts): {'PASS' if ok else 'FAIL'}**")
    txt = "\n".join(lines) + "\n"; (out / f"gate_report_{config}.md").write_text(txt); print(txt)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/week3"); ap.add_argument("--config", default="full")
    a = ap.parse_args(); report(ROOT / a.out, a.config)
