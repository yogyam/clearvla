"""Collect the H1 observation set: scripted expert drives unseen paired seeds; at every `--every`-th control step
(the observation points a policy with execute-8 would see) render both Blender views and the GT critical masks
for both views. Same frames for every acceleration method.

  python analysis/h1_collect.py --material opaque --seeds 1000-1049
Output: datasets/h1/<material>.h5 with images (N,2,224,224,3) uint8, masks (N,2,224,224) bool, and per-row meta.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(_v, "1")
import argparse, sys, time, json
from pathlib import Path
import numpy as np, h5py
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from PIL import Image
from envs.tasks import make_task
from experts.scripted import make_expert
from render.bridge import BlenderBridge, MaskRenderer
from eval.run_matrix import parse_seeds

VIEWS = ("front", "wrist")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True); ap.add_argument("--tasks", default="grasp,pour,insert")
    ap.add_argument("--seeds", default="1000-1049"); ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--out", default="datasets/h1")
    a = ap.parse_args(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp") / f"clearvla_h1_{a.material}_{os.getpid()}.png"
    rows = dict(task=[], seed=[], step=[], instruction_id=[], qpos=[], tcp_pos=[], tcp_R=[], gripper=[], fill=[], expert_success=[])
    imgs, masks = [], []; bridge = None; t0 = time.perf_counter(); n_ep = 0
    for task_name in a.tasks.split(","):
        task = make_task(task_name, a.material)
        for seed in parse_seeds(a.seeds):
            obs = task.reset(seed); ex = make_expert(task, seed)
            if bridge is None: bridge = BlenderBridge(task.model, task.info, a.material)
            bridge.rebind(task.model, task.info, a.material)
            mr = {v: MaskRenderer(task.model, task.info, camera=v) for v in VIEWS}
            done = False; idle = 0; ep_rows = []
            while not done:
                if task.t % a.every == 0:
                    im, mk = [], []
                    for v in VIEWS:
                        bridge.sync(task.data, camera=v); bridge.render(str(tmp)); im.append(np.asarray(Image.open(tmp).convert("RGB")))
                        crit, _ = mr[v](task.data); mk.append(crit)
                    imgs.append(np.stack(im)); masks.append(np.stack(mk)); ep_rows.append(len(imgs) - 1)
                    rows["task"].append(task_name); rows["seed"].append(seed); rows["step"].append(task.t); rows["instruction_id"].append(obs["instruction_id"])
                    rows["qpos"].append(obs["qpos"]); rows["tcp_pos"].append(obs["tcp_pos"]); rows["tcp_R"].append(obs["tcp_R"]); rows["gripper"].append(obs["gripper"]); rows["fill"].append(obs["fill"])
                act = ex(obs); obs, done = task.step(act)
                if ex.phase >= len(ex.phases):
                    idle += 1
                    if idle >= 5: break
            succ = bool(task.success()); rows["expert_success"].extend([succ] * len(ep_rows)); n_ep += 1
            print(f"{task_name} {a.material} seed {seed}: {len(ep_rows)} obs, success={succ}, {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    with h5py.File(out / f"{a.material}.h5", "w") as f:
        f.create_dataset("images", data=np.stack(imgs), compression="gzip", compression_opts=1)
        f.create_dataset("masks", data=np.stack(masks), compression="gzip", compression_opts=1)
        for k, v in rows.items():
            if k == "task": f.create_dataset(k, data=np.array(v, dtype="S8"))
            else: f.create_dataset(k, data=np.asarray(v))
        f.attrs.update(material=a.material, every=a.every, seeds=a.seeds, views=json.dumps(VIEWS), n_episodes=n_ep)
    print(f"done: {len(imgs)} observations from {n_ep} episodes in {(time.perf_counter()-t0)/60:.1f} min -> {out / (a.material + '.h5')}")
