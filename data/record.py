"""Stage 1: record paired scripted-expert demonstrations.

For each task, seeds 0,1,2,... are run under BOTH materials; an episode is kept only if it succeeds under
both, so the opaque and glass sets contain identical seeds (identical geometry, actions and masks).
Per control step we store everything needed to replay the scene in Blender later (full qpos, liquid levels),
plus the MuJoCo-alpha RGB and masks at the sampled steps.

  python data/record.py --per-cell 150 --stride 4 --workers 6
  python data/record.py --check
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(_v, "1")
import argparse, sys, json, time
from pathlib import Path
import multiprocessing as mp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
MATERIALS = ("opaque", "glass")


def record_episode(task_name, material, seed, stride, out_dir, exec_noise=0.0):
    import numpy as np, h5py, mujoco
    from envs.tasks import make_task
    from experts.scripted import make_expert
    from eval.runner import MujocoFrames
    task = make_task(task_name, material); obs = task.reset(seed); ex = make_expert(task, seed)
    frames = MujocoFrames(task.model, task.info); rng_noise = np.random.default_rng(seed + 30_000)
    T = dict(action=[], ctrl=[], qpos=[], qvel=[], qpos_full=[], tcp_pos=[], tcp_R=[], gripper=[], fill=[], liquid=[], attached=[], tube_pose=[])
    S = dict(step=[], rgb=[], mask_crit=[], mask_tube=[])
    done = False; idle = 0
    while not done:
        t = task.t
        if t % stride == 0:
            crit, tube = frames.masks(task.data)
            S["step"].append(t); S["rgb"].append(frames.rgb(task.data)); S["mask_crit"].append(crit); S["mask_tube"].append(tube)
        a = ex(obs)
        T["action"].append(a.copy())          # label = clean expert action
        if exec_noise > 0:                    # DART-style: execute a perturbed action so the recorded states cover recovery
            a = a.copy(); a[:3] += rng_noise.normal(0, exec_noise, 3); a[9] = float(np.clip(a[9] + rng_noise.normal(0, 0.05), 0, 1))
        T["action"][-1] = T["action"][-1]; T["qpos"].append(obs["qpos"]); T["qvel"].append(obs["qvel"]); T["qpos_full"].append(task.data.qpos.copy())
        T["tcp_pos"].append(obs["tcp_pos"]); T["tcp_R"].append(obs["tcp_R"]); T["gripper"].append(obs["gripper"]); T["fill"].append(obs["fill"])
        T["liquid"].append([task.liquid.source, task.liquid.receiver] if task.liquid else [0.0, 0.0])
        T["attached"].append(task.attached); T["tube_pose"].append(np.concatenate(task.tube_pose()[0:1] + (task.data.xquat[task.info.tube_body],)))
        obs, done = task.step(a)
        T["ctrl"].append(task.data.ctrl.copy())
        if ex.phase >= len(ex.phases):
            idle += 1
            if idle >= 5: break
    success = task.success()
    if not success: return dict(task=task_name, material=material, seed=seed, success=False, steps=task.t, failure=task.failure_reason())
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / f"ep_{seed:04d}.h5", "w") as f:
        f.attrs.update(task=task_name, material=material, seed=seed, instruction=task.instruction, instruction_id=task.instruction_id,
                       params=json.dumps(task.info.params), mark_level=float(task.info.mark_level), n_steps=task.t, stride=stride,
                       success=True, critical_geoms=json.dumps([int(g) for g in task.info.critical_geoms]), tube_geom=int(task.info.tube_geom), exec_noise=float(exec_noise))
        for k, v in T.items(): f.create_dataset(k, data=np.asarray(v, dtype=np.float32 if k != "attached" else bool))
        f.create_dataset("sample_steps", data=np.asarray(S["step"], np.int32))
        f.create_dataset("rgb_alpha", data=np.stack(S["rgb"]).astype(np.uint8), compression="gzip", compression_opts=1)
        f.create_dataset("mask_crit", data=np.stack(S["mask_crit"]), compression="gzip", compression_opts=1)
        f.create_dataset("mask_tube", data=np.stack(S["mask_tube"]), compression="gzip", compression_opts=1)
    return dict(task=task_name, material=material, seed=seed, success=True, steps=task.t, samples=len(S["step"]))


def _job(args):
    task_name, seed, stride, out_root, exec_noise = args
    res = [record_episode(task_name, m, seed, stride, Path(out_root) / f"{task_name}_{m}", exec_noise) for m in MATERIALS]
    return res


def check(out_root):
    root = Path(out_root); man = json.loads((root / "manifest.json").read_text())
    ok = True
    for t in man["tasks"]:
        for m in MATERIALS:
            n = len(list((root / f"{t}_{m}").glob("ep_*.h5"))); print(f"{t}/{m}: {n} episodes on disk, {len(man['kept'][t])} kept in manifest")
            ok &= n == len(man["kept"][t]) == man["per_cell"]
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="grasp,pour,insert"); ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="datasets/raw"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed-start", type=int, default=0); ap.add_argument("--append", action="store_true", help="extend an existing manifest instead of overwriting")
    ap.add_argument("--exec-noise", type=float, default=0.0, help="std (m) of Gaussian noise added to the EXECUTED xyz action; labels stay clean (DART)")
    a = ap.parse_args(); out_root = ROOT / a.out
    if a.check: check(out_root); sys.exit()
    tasks = a.tasks.split(","); t0 = time.perf_counter()
    manifest = dict(per_cell=a.per_cell, stride=a.stride, tasks=tasks, kept={}, rejected={}, steps={})
    old = json.loads((out_root / "manifest.json").read_text()) if (a.append and (out_root / "manifest.json").exists()) else None
    for task_name in tasks:
        kept, rejected, seed, batch = [], {}, a.seed_start, max(a.per_cell + 10, 40)
        while len(kept) < a.per_cell:
            jobs = [(task_name, s, a.stride, str(out_root), a.exec_noise) for s in range(seed, seed + batch)]
            with mp.get_context("spawn").Pool(a.workers) as pool: results = pool.map(_job, jobs, chunksize=2)
            for pair in results:
                s = pair[0]["seed"]
                if all(r["success"] for r in pair):
                    if len(kept) < a.per_cell: kept.append(s); manifest["steps"][f"{task_name}_{s}"] = pair[0]["steps"]
                    else:
                        for m in MATERIALS: (out_root / f"{task_name}_{m}" / f"ep_{s:04d}.h5").unlink(missing_ok=True)
                else:
                    rejected[s] = [r.get("failure", "") for r in pair]
                    for m in MATERIALS: (out_root / f"{task_name}_{m}" / f"ep_{s:04d}.h5").unlink(missing_ok=True)
            seed += batch; batch = max(a.per_cell - len(kept) + 5, 8)
            print(f"{task_name}: kept {len(kept)}/{a.per_cell}, rejected {len(rejected)}, next seed {seed}  ({time.perf_counter()-t0:.0f} s)", flush=True)
        manifest["kept"][task_name] = kept; manifest["rejected"][task_name] = rejected
    if old:
        for t in tasks:
            manifest["kept"][t] = old["kept"].get(t, []) + manifest["kept"][t]; manifest["rejected"][t] = {**old["rejected"].get(t, {}), **manifest["rejected"][t]}
        manifest["steps"] = {**old["steps"], **manifest["steps"]}; manifest["per_cell"] = old["per_cell"] + a.per_cell
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done in {time.perf_counter()-t0:.0f} s; manifest at {out_root/'manifest.json'}")
    check(out_root)
