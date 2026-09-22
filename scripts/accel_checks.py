"""Week 4 unit checks for accel/: full-budget variants reproduce `full`; action drift per method/budget; INT4 error; costs.
  python scripts/accel_checks.py --n 40
"""
import sys, argparse, json, time
from pathlib import Path
import numpy as np, torch, h5py
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from eval.policy import ModelAPolicy
from accel import make_accel, Accel

def load_obs(n, rng):
    """n observations from clean demos (both views), grouped so cache checks see consecutive frames of one episode."""
    out = []
    for task in ("grasp", "pour", "insert"):
        for mat in ("opaque", "glass"):
            files = sorted((ROOT / "datasets/raw" / f"{task}_{mat}").glob("ep_*.h5"))[-3:]
            for fp in files:
                with h5py.File(fp, "r") as f:
                    steps = f["sample_steps"][:]; sel = [i for i in range(len(steps)) if steps[i] % 8 == 0][:6]
                    for i in sel:
                        t = int(steps[i])
                        obs = dict(qpos=f["qpos"][t], tcp_pos=f["tcp_pos"][t], tcp_R=f["tcp_R"][t], gripper=float(f["gripper"][t]),
                                   instruction_id=int(f.attrs["instruction_id"]), instruction=str(f.attrs["instruction"]))
                        out.append((task, fp.name, t, [f["rgb_blender"][i], f["rgb_blender_wrist"][i]], obs))
    return out[:n] if n else out

def run(policy, ob, seed=0):
    task, _, _, imgs, obs = ob; torch.manual_seed(seed); return policy.observe(imgs, obs, task)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--ckpt", default="checkpoints/v4")
    a = ap.parse_args(); rng = np.random.default_rng(0); obs = load_obs(a.n, rng); print(f"{len(obs)} observations")
    base = ModelAPolicy(ROOT / a.ckpt); base.reset()
    ref = {}
    for i, ob in enumerate(obs): base.reset() if ob[1] != obs[i-1][1] else None; ref[i] = run(base, ob)
    report = {}
    for name in ["prune100", "cache100", "prune50", "prune25", "prune12", "cache50", "cache25", "cache12", "quant4", "protect_prune25", "protect_cache25"]:
        pol = ModelAPolicy(ROOT / a.ckpt, accel=make_accel(name)); pol.reset(); diffs = []; keeps = []; costs = []; recalls = []; t_obs = []
        for i, ob in enumerate(obs):
            if i == 0 or ob[1] != obs[i-1][1]: pol.reset()
            if pol.accel.needs_gt:                            # GT from the demo masks (front only in raw; wrist assumed empty here)
                with h5py.File(ROOT / "datasets/raw" / f"{ob[0]}_{'opaque' if 'opaque' in str(ob[1]) else 'glass'}" / ob[1], "r") as f: pass
                pol.accel.gt_crit = np.zeros(pol.n_vis, bool)   # protect path exercised with an empty mask; real masks come from the evaluator
            t0 = time.perf_counter(); acts = run(pol, ob); t_obs.append(time.perf_counter() - t0)
            d = np.abs(acts[:8, :3] - ref[i][:8, :3]).mean() * 1000; diffs.append(d)
            keeps.append(int(pol.diag[-1].sum())); costs.append(pol.costs[-1]["rel_to_full"])
        report[name] = dict(median_xyz_diff_mm=float(np.median(diffs)), p90_xyz_diff_mm=float(np.percentile(diffs, 90)), max_mm=float(np.max(diffs)),
                            mean_keep=float(np.mean(keeps)), rel_flops=float(np.mean(costs)), ms_per_obs=float(np.median(t_obs)) * 1000,
                            quant=pol.accel.quant_stats)
        print(name, json.dumps(report[name]))
    (ROOT / "results/week4").mkdir(exist_ok=True, parents=True)
    json.dump(report, open(ROOT / "results/week4/accel_checks.json", "w"), indent=1)
