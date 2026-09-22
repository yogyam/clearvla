"""Closed-loop matrix analysis (H2-H4): success per cell, retained success vs Full, diff-in-diff across materials with
paired bootstraps over seeds, failure modes, GT-patch recall and FLOPs per config.
  python analysis/matrix.py --out results/week5 > results/week5/matrix_report.md
"""
import sys, json, argparse, re, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
TASKS = ("grasp", "pour", "insert"); MATS = ("opaque", "glass")


def load(out, config, mat):
    p = ROOT / out / f"{config}_{mat}.jsonl"
    if not p.exists(): return {}
    d = {}
    for l in open(p):
        r = json.loads(l); d[(r["task"], r["seed"])] = r
    return d


def boot_mean(x, rng, n=10000):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (n, len(x))); return x[idx].mean(1)


def main(out, n_boot=10000):
    rng = np.random.default_rng(0)
    configs = sorted({re.sub(r"_(opaque|glass)\.jsonl$", "", p.name) for p in (ROOT / out).glob("*_opaque.jsonl")}, key=lambda c: (c != "full", c))
    data = {(c, m): load(out, c, m) for c in configs for m in MATS}
    lines = [f"# Closed-loop matrix — `{out}`\n", "Paired seeds; success = task check held 2 s; CIs = bootstrap over seeds (10k). Retained success = succ(config) / succ(full) per task × material; "
             "diff-in-diff = retained(opaque) − retained(glass), paired over seeds.\n", "## Success per cell (%, [95 % CI], n)\n",
             "| Config | " + " | ".join(f"{t} {m[0].upper()}" for t in TASKS for m in MATS) + " |", "|---|" + "---|" * 6]
    for c in configs:
        cells = []
        for t in TASKS:
            for m in MATS:
                rs = [r for (tt, s), r in data[(c, m)].items() if tt == t]
                if not rs: cells.append("—"); continue
                s = np.array([r["success"] for r in rs], float); b = boot_mean(s, rng); cells.append(f"{s.mean()*100:.0f} [{np.percentile(b,2.5)*100:.0f}, {np.percentile(b,97.5)*100:.0f}] n={len(s)}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    # retained success and diff-in-diff
    lines += ["\n## Retained success vs Full and material diff-in-diff (pts)\n", "| Config | Task | Retained opaque | Retained glass | DiD (opaque − glass) | 95 % CI |", "|---|---|---|---|---|---|"]
    verdict = {}
    for c in configs:
        if c == "full": continue
        for t in TASKS:
            seeds = sorted(set(s for (tt, s) in data[(c, "opaque")] if tt == t) & set(s for (tt, s) in data[(c, "glass")] if tt == t)
                           & set(s for (tt, s) in data[("full", "opaque")] if tt == t) & set(s for (tt, s) in data[("full", "glass")] if tt == t))
            if not seeds: continue
            S = {m: np.array([data[(c, m)][(t, s)]["success"] for s in seeds], float) for m in MATS}
            F = {m: np.array([data[("full", m)][(t, s)]["success"] for s in seeds], float) for m in MATS}
            idx = rng.integers(0, len(seeds), (n_boot, len(seeds)))
            def retained(m, ix=None):
                sm, fm = (S[m], F[m]) if ix is None else (S[m][ix], F[m][ix]); fmean = fm.mean(-1); return np.where(fmean > 0, sm.mean(-1) / np.maximum(fmean, 1e-9), np.nan)
            ro, rg = retained("opaque"), retained("glass"); did = (ro - rg) * 100
            bo = (retained("opaque", idx) - retained("glass", idx)) * 100; lo, hi = np.nanpercentile(bo, [2.5, 97.5])
            verdict[(c, t)] = (did, lo, hi, ro, rg)
            lines.append(f"| {c} | {t} | {ro*100:.0f} % | {rg*100:.0f} % | {did:+.0f} | [{lo:+.0f}, {hi:+.0f}] |")
    # failure modes, recall, flops, latency
    lines += ["\n## Failure modes, GT-patch recall, compute\n", "| Config | Material | Failures (top 3) | Mean GT recall | GFLOPs | Policy ms/obs |", "|---|---|---|---|---|---|"]
    for c in configs:
        for m in MATS:
            rs = list(data[(c, m)].values())
            if not rs: continue
            h = collections.Counter(re.sub(r"[\d.]+", "#", r["failure"]) for r in rs if not r["success"]).most_common(3)
            ks = [k for r in rs for k in r.get("keep_sets", [])]; rec = np.nanmean([k["recall"] for k in ks if k["n_crit"] > 0]) if ks else float("nan")
            gf = np.mean([r["gflops"] for r in rs if r.get("gflops")]) if any(r.get("gflops") for r in rs) else float("nan")
            lines.append(f"| {c} | {m} | {', '.join(f'{k}×{v}' for k, v in h)} | {'—' if np.isnan(rec) else f'{rec*100:.0f} %'} | {'—' if np.isnan(gf) else f'{gf:.1f}'} | {np.mean([r['t_policy'] for r in rs])*1000:.0f} |")
    # verdicts
    lines.append("\n## Pre-registered verdicts\n")
    for c in configs:
        if c == "full": continue
        vs = [verdict[(c, t)] for t in TASKS if (c, t) in verdict]
        hits = [t for t in TASKS if (c, t) in verdict and verdict[(c, t)][0] >= 15 and verdict[(c, t)][1] > 0]
        if c.startswith("quant"):
            gaps = [f"{t}: {verdict[(c,t)][0]:+.0f}" for t in TASKS if (c, t) in verdict]; ok = all(abs(verdict[(c, t)][0]) < 5 for t in TASKS if (c, t) in verdict)
            lines.append(f"- **H4 (Quant material-neutral, |DiD| < 5 pts):** {', '.join(gaps)} → {'supported' if ok else 'not supported (still interesting)'}")
        else:
            lines.append(f"- **H2 for {c}** (≥ 15 pts lower retained success on glass, CI excludes 0, on ≥ 2 of 3 tasks): tasks meeting it = {hits or 'none'} → {'SUPPORTED' if len(hits) >= 2 else 'not supported'}")
    def drop(c, t): v = verdict.get((c, t)); return None if v is None else 1 - (v[3] + v[4]) / 2
    for c in configs:
        if c in ("full",) or c.startswith("quant") or c.startswith("protect"): continue
        d = {t: drop(c, t) for t in TASKS}; d = {t: v for t, v in d.items() if v is not None}
        if d: worst = max(d, key=d.get); lines.append(f"- **H3 ({c} hurts which task most, mean retained-success loss):** " + ", ".join(f"{t} {v*100:.0f} pts" for t, v in d.items()) + f" → worst = {worst} (pre-registered: Cache → pour, Prune → grasp)")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/week5"); a = ap.parse_args(); main(a.out)
