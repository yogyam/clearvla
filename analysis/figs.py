"""Paper figures -> results/figs/. Run in the clearvla env.
  python analysis/figs.py --figs 3,5        # log-only figures (no GPU)
  python analysis/figs.py --figs 1,4        # need Model A on MPS (a few forward passes)
"""
import sys, json, argparse, re, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); OUT = ROOT / "results/figs"; OUT.mkdir(parents=True, exist_ok=True)
TASKS = ("grasp", "pour", "insert"); MATS = ("opaque", "glass"); COL = {"opaque": "#444444", "glass": "#1f77b4"}


def fig3_h1_recall():
    rows = [json.loads(l) for m in MATS for l in open(ROOT / "results/week4" / f"h1_{m}.jsonl")]
    budgets = [("12", 12.5), ("25", 25), ("50", 50)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, meth in zip(axes, ("prune", "cache")):
        for m in MATS:
            means, los, his = [], [], []
            for b, _ in budgets:
                ep = collections.defaultdict(list)
                for r in rows:
                    if r["material"] != m: continue
                    v = r["recall"][f"{meth}{b}"]["all"]
                    if np.isnan(v) or (meth == "cache" and r["first"]): continue
                    ep[(r["task"], r["seed"])].append(v)
                e = np.array([np.mean(v) for v in ep.values()]) * 100; rng = np.random.default_rng(0)
                bs = e[rng.integers(0, len(e), (5000, len(e)))].mean(1); means.append(e.mean()); los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
            x = [b for _, b in budgets]; ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - means], marker="o", color=COL[m], label=m, capsize=3)
        ax.plot([12.5, 25, 50], [12.5, 25, 50], "k:", lw=1, label="chance" if meth == "prune" else None)
        ax.set_xscale("log", base=2); ax.set_xticks([12.5, 25, 50]); ax.set_xticklabels(["12.5", "25", "50"]); ax.set_xlabel("budget (% of visual tokens)")
        ax.set_title({"prune": "Prune: attention-ranked keep set", "cache": "Cache: pixel-change recompute set"}[meth]); ax.grid(alpha=.3)
    axes[0].set_ylabel("critical-patch recall (%)"); axes[0].legend(); fig.tight_layout(); fig.savefig(OUT / "fig3_h1_recall.png", dpi=200); plt.close(fig); print("fig3 written")


def _load(out, c, m):
    p = ROOT / out / f"{c}_{m}.jsonl"; return [json.loads(l) for l in open(p)] if p.exists() else []


def fig2_budget_curve():
    """Success vs budget per task and material, Full at 100 %, from results/week5 + results/week6."""
    cfgs = {"prune": [("prune12", 12.5), ("prune25", 25), ("prune50", 50)], "cache": [("cache25", 25), ("cache50", 50)]}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, t in zip(axes, TASKS):
        for m in MATS:
            full = [r for r in _load("results/week5", "full", m) if r["task"] == t]; f = np.mean([r["success"] for r in full]) * 100
            for meth, pts in cfgs.items():
                xs, ys = [], []
                for c, b in pts:
                    rs = [r for out in ("results/week5", "results/week6") for r in _load(out, c, m) if r["task"] == t]
                    if rs: xs.append(b); ys.append(np.mean([r["success"] for r in rs]) * 100)
                if xs: ax.plot(xs + [100], ys + [f], marker="o" if meth == "prune" else "s", ls="-" if meth == "prune" else "--", color=COL[m], label=f"{meth} {m}")
            q = [r for r in _load("results/week5", "quant4", m) if r["task"] == t]
            if q: ax.scatter([100], [np.mean([r["success"] for r in q]) * 100], marker="*", s=90, color=COL[m], zorder=5, label=f"INT4 {m}")
        ax.set_xscale("log", base=2); ax.set_xticks([12.5, 25, 50, 100]); ax.set_xticklabels(["12.5", "25", "50", "full"]); ax.set_title(t); ax.set_xlabel("budget (%)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("closed-loop success (%)"); axes[-1].legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "fig2_budget_curve.png", dpi=200); plt.close(fig); print("fig2 written")


def fig5_prune_failures():
    """Failure modes and final xy offsets, Full vs Prune-25, per material (Week 5)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    cats = ["ok", "xy_off", "tube_knocked_over", "spilled", "underfilled", "overfilled", "other"]
    def cat(r):
        if r["success"]: return "ok"
        for c in cats[1:-1]:
            if r["failure"].startswith(c): return c
        return "other"
    width = 0.2; x = np.arange(len(cats))
    for i, (c, m) in enumerate([(c, m) for c in ("full", "prune25") for m in MATS]):
        rs = [r for r in _load("results/week5", c, m) if r["task"] == "grasp"]; h = collections.Counter(cat(r) for r in rs)
        axes[0].bar(x + (i - 1.5) * width, [h[k] / max(len(rs), 1) * 100 for k in cats], width, color=COL[m], alpha=1 if c == "full" else .5, label=f"{c} {m}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats, rotation=30, ha="right", fontsize=8); axes[0].set_ylabel("% of grasp episodes"); axes[0].legend(fontsize=7); axes[0].set_title("Grasp outcome, Full vs Prune-25")
    for c, ls in (("full", "-"), ("prune25", "--")):
        for m in MATS:
            offs = [int(re.search(r"(\d+)mm", r["failure"]).group(1)) for r in _load("results/week5", c, m) if r["task"] == "grasp" and "xy_off" in r["failure"]]
            if offs: axes[1].hist(offs, bins=np.linspace(0, 300, 16), histtype="step", ls=ls, color=COL[m], label=f"{c} {m} (n={len(offs)})")
    axes[1].set_xlabel("final tube offset from slot (mm), failed grasp episodes"); axes[1].set_ylabel("episodes"); axes[1].legend(fontsize=7); axes[1].set_title("Placement misses")
    fig.tight_layout(); fig.savefig(OUT / "fig5_prune_failures.png", dpi=200); plt.close(fig); print("fig5 written")


def fig1_keep_sets(seed=1003):
    """Same scene, opaque and glass, both views, Prune-25 keep set (yellow) and GT critical patches (red outline)."""
    import torch, h5py
    from eval.policy import ModelAPolicy
    from accel import make_accel
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for i, m in enumerate(MATS):
        with h5py.File(ROOT / "datasets/h1" / f"{m}.h5", "r") as f:
            tasks = [t.decode() for t in f["task"][:]]; seeds = f["seed"][:]; steps = f["step"][:]
            idx = [k for k in range(len(tasks)) if tasks[k] == "grasp" and seeds[k] == seed and steps[k] == 24][0]
            imgs = [f["images"][idx][0], f["images"][idx][1]]; masks = f["masks"][idx]
            obs = dict(qpos=f["qpos"][idx], tcp_pos=f["tcp_pos"][idx], tcp_R=f["tcp_R"][idx], gripper=float(f["gripper"][idx]), instruction_id=int(f["instruction_id"][idx]), instruction="")
        pol = ModelAPolicy(ROOT / "checkpoints/v4", accel=make_accel("prune25")); pol.reset(); torch.manual_seed(0); pol.observe(imgs, obs, "grasp"); keep = pol.diag[-1]
        for j, v in enumerate(("front", "wrist")):
            ax = axes[i, j]; ax.imshow(imgs[j]); k = keep[j * 196:(j + 1) * 196].reshape(14, 14); crit = masks[j].reshape(14, 16, 14, 16).mean((1, 3)) >= 0.25
            over = np.zeros((14, 14, 4)); over[k] = [1, 0.9, 0, 0.35]; ax.imshow(np.kron(over, np.ones((16, 16, 1))))
            for (r, c) in zip(*np.nonzero(crit)): ax.add_patch(plt.Rectangle((c * 16, r * 16), 16, 16, fill=False, ec="red", lw=1.5))
            ax.set_title(f"{m}, {v}: kept {int(k.sum())}/196, object patches kept {int((k & crit).sum())}/{int(crit.sum())}", fontsize=8); ax.axis("off")
    fig.suptitle("Prune-25 keep set (yellow) vs GT object patches (red), same scene/seed", fontsize=10); fig.tight_layout(); fig.savefig(OUT / "fig1_keep_sets.png", dpi=200); plt.close(fig); print("fig1 written")


def fig4_cache_staleness():
    """Static pixels are not static tokens: pixel similarity vs SigLIP feature cosine between consecutive observations;
    action drift vs observation index for Cache-50/25 along expert episodes."""
    import torch, h5py
    from eval.policy import ModelAPolicy
    from accel import make_accel
    from accel.cache import patch_similarity
    pol_full = ModelAPolicy(ROOT / "checkpoints/v4"); ps_all, fs_all = [], []; drifts = {"cache50": [], "cache25": []}
    with h5py.File(ROOT / "datasets/h1/opaque.h5", "r") as f:
        tasks = [t.decode() for t in f["task"][:]]; seeds = f["seed"][:]
        def obs_at(i): return [f["images"][i][0], f["images"][i][1]], dict(qpos=f["qpos"][i], tcp_pos=f["tcp_pos"][i], tcp_R=f["tcp_R"][i], gripper=float(f["gripper"][i]), instruction_id=int(f["instruction_id"][i]), instruction="")
        for seed in (1000, 1001, 1002, 1003):
            ii = [i for i in range(len(tasks)) if tasks[i] == "grasp" and seeds[i] == seed]
            feats = []
            for i in ii:
                imgs, _ = obs_at(i)
                with torch.no_grad(): feats.append(torch.cat([pol_full.enc(im) for im in imgs], 1)[0].float().cpu().numpy())
            for a, b in zip(ii[:-1], ii[1:]):
                pa, pb = obs_at(a)[0], obs_at(b)[0]; ps_all.append(np.concatenate([patch_similarity(x, y) for x, y in zip(pa, pb)]))
                fa, fb = feats[ii.index(a)], feats[ii.index(b)]; fs_all.append((fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1)))
            for cfg in drifts:
                pol = ModelAPolicy(ROOT / "checkpoints/v4", accel=make_accel(cfg)); pol.reset(); pol_full.reset(); d = []
                for i in ii:
                    imgs, o = obs_at(i); torch.manual_seed(0); af = pol_full.observe(imgs, o, "grasp"); torch.manual_seed(0); ac = pol.observe(imgs, o, "grasp"); d.append(np.abs(af[:8, :3] - ac[:8, :3]).mean() * 1000)
                drifts[cfg].append(d)
    ps, fs = np.concatenate(ps_all), np.concatenate(fs_all)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].scatter(ps, fs, s=2, alpha=.15, color="#333"); axes[0].axvline(0.995, color="red", ls=":", lw=1); axes[0].set_xlabel("pixel similarity of the patch to the previous observation"); axes[0].set_ylabel("SigLIP token cosine to previous observation")
    st = ps > 0.995; axes[0].set_title(f"'static' patches (right of the line, {st.mean()*100:.0f} %): token cosine {fs[st].mean():.2f} mean, p10 {np.percentile(fs[st],10):.2f}", fontsize=8); axes[0].grid(alpha=.3)
    for cfg, ls in (("cache50", "-"), ("cache25", "--")):
        L = min(len(d) for d in drifts[cfg]); arr = np.array([d[:L] for d in drifts[cfg]]); axes[1].plot(range(L), arr.mean(0), ls=ls, marker="o", ms=3, label=f"{cfg} (mean of 4 episodes)")
    axes[1].set_xlabel("observation index in the episode (every 8 control steps)"); axes[1].set_ylabel("|action − Full| (mm, first 8 steps)"); axes[1].legend(); axes[1].grid(alpha=.3); axes[1].set_title("Action drift from stale cached tokens", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig4_cache_staleness.png", dpi=200); plt.close(fig); print("fig4 written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--figs", default="2,3,5"); a = ap.parse_args()
    for f in a.figs.split(","):
        {"1": fig1_keep_sets, "2": fig2_budget_curve, "3": fig3_h1_recall, "4": fig4_cache_staleness, "5": fig5_prune_failures}[f]()
