"""H1: critical-patch recall of each acceleration method's keep set on the H1 observation set (analysis/h1_collect.py).

  python analysis/h1_recall.py --material opaque      # -> results/week4/h1_opaque.jsonl
  python analysis/h1_recall.py --report               # -> results/week4/h1_report.md (bootstrap over paired episodes)

Keep set = visual tokens (both views, 392) that get full prefix compute after layer 2: Prune keeps the top-b by
layer-2 attention from text+proprio; Cache recomputes the b most pixel-changed since the previous observation
(first observation of an episode: all). Critical patch = >= 25 % of its pixels on the instruction-referenced object
(simulator segmentation). Recall = |keep ∩ crit| / |crit|, pooled over both views (and per view).
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, torch, h5py
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
BUDGETS = {"50": 0.5, "25": 0.25, "12": 0.125}


def crit_bits(mask_views):                     # (2,224,224) bool -> (392,) bool
    return np.concatenate([(m.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.25).flatten() for m in mask_views])


def recall(keep, crit):
    n = int(crit.sum()); return float((keep & crit).sum() / n) if n else float("nan")


def compute(material, ckpt="checkpoints/v4"):
    from eval.policy import ModelAPolicy
    from data.dataset import normalise_prop, tcp10
    from accel.prune import prune_scores, topk_mask
    from accel.cache import patch_similarity
    pol = ModelAPolicy(ROOT / ckpt); n_vis = pol.n_vis; dev = pol.dev
    with h5py.File(ROOT / "datasets/h1" / f"{material}.h5", "r") as f:
        images, masks = f["images"], f["masks"]; tasks = [t.decode() for t in f["task"][:]]; seeds = f["seed"][:]; steps = f["step"][:]
        iid = f["instruction_id"][:]; qpos = f["qpos"][:]; tcp_pos = f["tcp_pos"][:]; tcp_R = f["tcp_R"][:]; grip = f["gripper"][:]; succ = f["expert_success"][:]
        out = open(ROOT / "results/week4" / f"h1_{material}.jsonl", "w"); prev = None
        for i in range(len(tasks)):
            imgs = images[i]; crit = crit_bits(masks[i])
            with torch.no_grad():
                vis = torch.cat([pol.enc(im) for im in imgs], 1)
                row, L = pol.text_row[(tasks[i], int(iid[i]))]
                txt = torch.from_numpy(pol.text[row])[None].to(dev); txt_mask = torch.zeros(1, 64, dtype=torch.bool, device=dev); txt_mask[0, :L] = True
                prop = np.concatenate([qpos[i], tcp10(tcp_pos[i], tcp_R[i], np.float32(grip[i]))]).astype(np.float32)
                prop_t = torch.from_numpy(normalise_prop(prop, pol.norm))[None].to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16):
                    _, key_mask, _, attns = pol.model.prefix(vis, txt, txt_mask, prop_t, return_attn=True)
                scores = prune_scores(attns[2].float(), n_vis, key_mask)[0].cpu().numpy()
            same_ep = prev is not None and prev[0] == (tasks[i], int(seeds[i]))
            sim = np.concatenate([patch_similarity(p, c) for p, c in zip(prev[1], imgs)]) if same_ep else None
            rec = {}
            for name, b in BUDGETS.items():
                k = int(round(b * n_vis))
                kp = topk_mask(torch.from_numpy(scores)[None], k)[0].numpy()
                if sim is None: kc = np.ones(n_vis, bool)
                else: kc = np.zeros(n_vis, bool); kc[np.argsort(sim)[:k]] = True
                for m, kk in (("prune", kp), ("cache", kc)):
                    rec[f"{m}{name}"] = dict(all=recall(kk, crit), front=recall(kk[:196], crit[:196]), wrist=recall(kk[196:], crit[196:]))
            attn_mass = float(scores[crit].sum() / scores.sum()) if crit.any() else float("nan")
            out.write(json.dumps(dict(material=material, task=tasks[i], seed=int(seeds[i]), step=int(steps[i]), first=not same_ep,
                                      n_crit_front=int(crit[:196].sum()), n_crit_wrist=int(crit[196:].sum()), attn_mass_crit=attn_mass,
                                      sim_front=float(np.mean(sim[:196])) if sim is not None else None, sim_wrist=float(np.mean(sim[196:])) if sim is not None else None,
                                      expert_success=bool(succ[i]), recall=rec)) + "\n")
            prev = ((tasks[i], int(seeds[i])), [np.asarray(im).copy() for im in imgs])
            if i % 500 == 0: print(f"{material}: {i}/{len(tasks)}", flush=True)
        out.close(); print(f"{material}: done {len(tasks)} observations")


def report(n_boot=10000):
    rows = [json.loads(l) for m in ("opaque", "glass") for l in open(ROOT / "results/week4" / f"h1_{m}.jsonl")]
    methods = [f"{m}{b}" for m in ("prune", "cache") for b in BUDGETS]
    # episode-level mean recall (over observations with >= 1 critical patch; cache: exclude the first observation)
    def ep_means(view="all"):
        d = {}
        for r in rows:
            for mth in methods:
                v = r["recall"][mth][view]
                if np.isnan(v) or (mth.startswith("cache") and r["first"]): continue
                d.setdefault((mth, r["task"], r["material"], r["seed"]), []).append(v)
        return {k: float(np.mean(v)) for k, v in d.items()}
    rng = np.random.default_rng(0); lines = ["# H1 — critical-patch recall of the keep sets (offline, expert rollouts, seeds 1000–1049)\n",
        "Keep set = visual tokens with full prefix compute after layer 2 (both views, 392 tokens). Critical patch = ≥ 25 % of its pixels on the "
        "instruction-referenced object (simulator GT). Recall = |keep ∩ crit| / |crit|. Episode means, then mean over episodes; gap = opaque − glass "
        "with a bootstrap over paired seeds (10k resamples). Cache excludes the first observation of each episode (always full).\n"]
    for view in ("all", "front", "wrist"):
        em = ep_means(view); lines.append(f"\n## Recall, view = {view}\n\n| Method | Task | Opaque | Glass | Gap (pts) | 95 % CI | n seeds |\n|---|---|---|---|---|---|---|")
        for mth in methods:
            for task in ("grasp", "pour", "insert", "pooled"):
                tasks = ("grasp", "pour", "insert") if task == "pooled" else (task,)
                pairs = [(em[(mth, t, "opaque", s)], em[(mth, t, "glass", s)]) for t in tasks for s in range(1000, 1050) if (mth, t, "opaque", s) in em and (mth, t, "glass", s) in em]
                if not pairs: continue
                o, g = np.array(pairs).T * 100; gap = o.mean() - g.mean()
                idx = rng.integers(0, len(pairs), (n_boot, len(pairs))); boots = (o[idx] - g[idx]).mean(1); lo, hi = np.percentile(boots, [2.5, 97.5])
                lines.append(f"| {mth} | {task} | {o.mean():.1f} | {g.mean():.1f} | {gap:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {len(pairs)} |")
    # mechanism: attention mass on critical tokens, pixel similarity per view
    lines.append("\n## Mechanism\n\n| Task | Material | Attention mass on critical tokens (layer 2) | Critical patches front / wrist | Pixel similarity to previous obs, front / wrist |\n|---|---|---|---|---|")
    for task in ("grasp", "pour", "insert"):
        for mat in ("opaque", "glass"):
            rr = [r for r in rows if r["task"] == task and r["material"] == mat]
            am = np.nanmean([r["attn_mass_crit"] for r in rr]); nf = np.mean([r["n_crit_front"] for r in rr]); nw = np.mean([r["n_crit_wrist"] for r in rr])
            sf = np.mean([r["sim_front"] for r in rr if r["sim_front"] is not None]); sw = np.mean([r["sim_wrist"] for r in rr if r["sim_wrist"] is not None])
            lines.append(f"| {task} | {mat} | {am*100:.1f} % | {nf:.1f} / {nw:.1f} | {sf:.3f} / {sw:.3f} |")
    em = ep_means("all")
    verdicts = []
    for mth in ("prune25", "cache25"):
        pairs = [(em[(mth, t, "opaque", s)], em[(mth, t, "glass", s)]) for t in ("grasp", "pour", "insert") for s in range(1000, 1050) if (mth, t, "opaque", s) in em and (mth, t, "glass", s) in em]
        o, g = np.array(pairs).T * 100; gap = o.mean() - g.mean(); idx = rng.integers(0, len(pairs), (n_boot, len(pairs))); lo, hi = np.percentile((o[idx] - g[idx]).mean(1), [2.5, 97.5])
        ok = gap >= 10 and lo > 0; verdicts.append(f"- **{mth}** pooled gap {gap:+.1f} pts [{lo:+.1f}, {hi:+.1f}] → {'SUPPORTED' if ok else 'not supported'} (needs ≥ +10 with CI excluding 0)")
    lines.append("\n## H1 verdict (pre-registered: recall(opaque) − recall(glass) ≥ 10 pts at the 25 % budget, CI excludes 0)\n\n" + "\n".join(verdicts))
    (ROOT / "results/week4/h1_report.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--material"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.material: compute(a.material)
    if a.report: report()
