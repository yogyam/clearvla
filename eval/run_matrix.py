"""Paired closed-loop evaluation of a policy config on Blender-rendered observations.

  python eval/run_matrix.py --ckpt checkpoints/full --config full --material glass --seeds 1000-1099 --out results/week3
One process per material (bpy is single-process). Logs one JSON line per episode; resumable (skips logged seeds).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from envs.tasks import make_task, HORIZON
from render.bridge import BlenderBridge, MaskRenderer
from eval.policy import ModelAPolicy
from PIL import Image


def parse_seeds(s):
    a, b = s.split("-"); return list(range(int(a), int(b) + 1))


def run(policy, task, bridge, masks, seed, tmp, strip_dir=None, strip_every=16):
    obs = task.reset(seed); policy.reset(); bridge.rebind(task.model, task.info, task.material)
    masks = MaskRenderer(task.model, task.info)
    t0 = time.perf_counter(); done = False; keep_sets = []; frames = []; held = 0
    while not done:
        if policy.need_observation():
            bridge.sync(task.data); bridge.render(str(tmp)); img = np.asarray(Image.open(tmp).convert("RGB"))
            policy.observe(img, obs, task.name)
            if policy.diag: crit, _ = masks(task.data); keep_sets.append(dict(step=task.t, keep=policy.diag[-1].tolist(), crit_patches=(crit.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.25).flatten().tolist()))
            if strip_dir is not None and task.t % strip_every == 0: frames.append(img)
        obs, done = task.step(policy.next_action())
        held = held + 1 if task.success() else 0
        if held >= 20: break                      # success sustained for 2 s: terminate early
    succ = task.success()
    if strip_dir is not None and frames:
        Image.fromarray(np.concatenate(frames[:12], 1)).save(strip_dir / f"{task.name}_{task.material}_s{seed}_{'ok' if succ else 'fail'}.png")
    return dict(task=task.name, material=task.material, seed=seed, success=bool(succ), steps=task.t, failure="" if succ else task.failure_reason(),
                fill=float(task.fill_level()), mark=float(task.info.mark_level), n_obs=len(policy.timings),
                t_encoder=float(np.mean([x["encoder"] for x in policy.timings])), t_policy=float(np.mean([x["policy"] for x in policy.timings])),
                wall=time.perf_counter() - t0, keep_sets=keep_sets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", default="full"); ap.add_argument("--material", required=True)
    ap.add_argument("--tasks", default="grasp,pour,insert"); ap.add_argument("--seeds", default="1000-1099"); ap.add_argument("--out", default="results/week3")
    ap.add_argument("--exec", type=int, default=8); ap.add_argument("--strips", type=int, default=3)
    a = ap.parse_args(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); strip_dir = out / "rollouts"; strip_dir.mkdir(exist_ok=True)
    log_path = out / f"{a.config}_{a.material}.jsonl"
    done = {(r["task"], r["seed"]) for r in (json.loads(l) for l in log_path.read_text().splitlines())} if log_path.exists() else set()
    accel = {}
    if a.config != "full":
        from accel import make_accel; accel = make_accel(a.config)
    policy = ModelAPolicy(ROOT / a.ckpt, exec_horizon=a.exec, accel=accel)
    tmp = Path("/tmp") / f"clearvla_eval_{a.material}_{os.getpid()}.png"; bridge = None
    seeds = parse_seeds(a.seeds); t_start = time.perf_counter(); n = 0
    with open(log_path, "a") as log:
        for task_name in a.tasks.split(","):
            task = make_task(task_name, a.material); task.reset(seeds[0])
            if bridge is None: bridge = BlenderBridge(task.model, task.info, a.material)
            for k, seed in enumerate(seeds):
                if (task_name, seed) in done: continue
                rec = run(policy, task, bridge, None, seed, tmp, strip_dir if k < a.strips else None)
                log.write(json.dumps(rec) + "\n"); log.flush(); n += 1
                if n % 10 == 0:
                    rows = [json.loads(l) for l in log_path.read_text().splitlines()]
                    cur = [r for r in rows if r["task"] == task_name]
                    print(f"[{a.config}/{a.material}] {task_name}: {sum(r['success'] for r in cur)}/{len(cur)}  ({(time.perf_counter()-t_start)/60:.1f} min, {rec['wall']:.1f} s/ep)", flush=True)
    print(f"done {n} new episodes in {(time.perf_counter()-t_start)/60:.1f} min -> {log_path}", flush=True)
