"""Week 1 exit check: scripted-expert success over N seeds x 3 tasks x materials (state-only tier),
with failure triage. Optionally renders review frames through Blender.

  python scripts/verify_experts.py --trials 100                       # success table -> results/week1/expert_success.md
  python scripts/verify_experts.py --render --frames 20 --trials 2    # Blender frames -> results/week1/frames/
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(_v, "1")
import argparse, sys, json, time, collections
from pathlib import Path
import multiprocessing as mp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def _run(args):
    name, material, seed = args
    from envs.tasks import make_task
    from experts.scripted import make_expert
    from eval.runner import run_episode
    task = make_task(name, material); task.reset(seed); ex = make_expert(task, seed)
    log = run_episode(task, ex, seed)
    return dict(task=name, material=material, seed=seed, success=log.success, steps=log.steps, failure=log.failure, wall=log.wall_s)


def verify(trials, tasks, materials, workers, out_dir):
    jobs = [(t, m, s) for t in tasks for m in materials for s in range(trials)]
    t0 = time.perf_counter()
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool: rows = pool.map(_run, jobs, chunksize=4)
    else:
        rows = [_run(j) for j in jobs]
    wall = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expert_episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    lines = [f"# Scripted expert verification ({trials} seeds per cell, {wall:.0f} s wall, {workers} workers)\n",
             "| Task | Material | Success | Mean steps | Failures |", "|---|---|---|---|---|"]
    ok_all = True
    for t in tasks:
        for m in materials:
            cell = [r for r in rows if r["task"] == t and r["material"] == m]
            n_ok = sum(r["success"] for r in cell); rate = n_ok / len(cell)
            fails = collections.Counter(r["failure"].split("_")[0] if r["failure"] else "" for r in cell if not r["success"])
            ftxt = ", ".join(f"{k}×{v}" for k, v in fails.most_common()) or "—"
            steps = sum(r["steps"] for r in cell) / len(cell)
            ok_all &= rate >= 0.95
            lines.append(f"| {t} | {m} | **{100*rate:.0f}%** ({n_ok}/{len(cell)}) | {steps:.0f} | {ftxt} |")
    # paired-seed consistency: expert is state based, so opaque and glass must agree on every seed
    mism = [(r["task"], r["seed"]) for r in rows if r["material"] == materials[0]
            for r2 in rows if r2["task"] == r["task"] and r2["seed"] == r["seed"] and r2["material"] != r["material"] and r2["success"] != r["success"]]
    lines.append(f"\nPaired-seed mismatches across materials: {len(mism)} {mism[:10]}")
    lines.append(f"\n**Gate (≥95% every cell): {'PASS' if ok_all else 'FAIL'}**")
    failed_seeds = [f"{r['task']}/{r['material']}/s{r['seed']}: {r['failure']}" for r in rows if not r["success"]]
    if failed_seeds: lines.append("\nFailed episodes:\n" + "\n".join(f"- {x}" for x in failed_seeds[:60]))
    (out_dir / "expert_success.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def render_frames(tasks, materials, n_seeds, frames, out_dir):
    import mujoco
    from envs.tasks import make_task
    from experts.scripted import make_expert
    from eval.runner import run_episode
    from render.bridge import BlenderBridge
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge = None
    for t in tasks:
        for m in materials:
            for seed in range(n_seeds):
                task = make_task(t, m); task.reset(seed); ex = make_expert(task, seed)
                if bridge is None: bridge = BlenderBridge(task.model, task.info, m)
                else: bridge.rebind(task.model, task.info, m)
                every = max(1, 300 // frames)
                log = run_episode(task, ex, seed, render="blender", bridge=bridge, frames_dir=out_dir, frame_every=every)
                print(f"{t}/{m}/s{seed}: success={log.success} steps={log.steps} frames every {every}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100); ap.add_argument("--tasks", default="grasp,pour,insert")
    ap.add_argument("--materials", default="opaque,glass"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--render", action="store_true"); ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="results/week1")
    a = ap.parse_args()
    tasks, mats = a.tasks.split(","), a.materials.split(",")
    if a.render: render_frames(tasks, mats, a.trials, a.frames, ROOT / a.out / "frames")
    else: verify(a.trials, tasks, mats, a.workers, ROOT / a.out)
