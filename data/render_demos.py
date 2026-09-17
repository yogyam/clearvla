"""Stage 2: render Blender (Cycles) frames for every sampled step of every recorded episode.

Replays each episode from the stored full qpos + liquid levels (scene rebuilt deterministically from the seed),
so no physics is re-run. Writes /rgb_blender (S,224,224,3) into each HDF5. Resumable: episodes that already
have a complete /rgb_blender are skipped. One bpy process per invocation; run one material at a time.

  python data/render_demos.py --material glass --resume
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, h5py, mujoco
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from envs.scene import build_scene, home, set_liquid_height, TUBE_H, TUBE_WALL, BEAKER_H
from render.bridge import BlenderBridge
from PIL import Image


def replay_frame(model, data, info, qpos_full, liquid):
    data.qpos[:] = qpos_full
    if info.source_liquid_geom >= 0:
        set_liquid_height(model, info.source_liquid_geom, -TUBE_H / 2 + TUBE_WALL, liquid[0] * (TUBE_H - 2 * TUBE_WALL))
        set_liquid_height(model, info.receiver_liquid_geom, info.beaker_center[2] - model.body_pos[info.fixture_body][2], liquid[1] * BEAKER_H)
    mujoco.mj_forward(model, data)


def render_episode(bridge, path: Path, tmp: Path, samples: int, camera: str = "front"):
    key = "rgb_blender" if camera == "front" else f"rgb_blender_{camera}"
    with h5py.File(path, "r") as f:
        task, material, seed = f.attrs["task"], f.attrs["material"], int(f.attrs["seed"])
        steps = f["sample_steps"][:]; qpos = f["qpos_full"][:]; liquid = f["liquid"][:]
        if key in f and f[key].shape[0] == len(steps): return 0
    model, info = build_scene(task, material, seed); data = mujoco.MjData(model); home(model, data)
    bridge.rebind(model, info, material)
    out = np.zeros((len(steps), bridge.res, bridge.res, 3), np.uint8)
    for i, t in enumerate(steps):
        replay_frame(model, data, info, qpos[t], liquid[t]); bridge.sync(data, camera=camera); bridge.render(str(tmp))
        out[i] = np.asarray(Image.open(tmp).convert("RGB"))
    with h5py.File(path, "a") as f:
        if key in f: del f[key]
        f.create_dataset(key, data=out, compression="gzip", compression_opts=1)
    return len(steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True); ap.add_argument("--tasks", default="grasp,pour,insert")
    ap.add_argument("--raw", default="datasets/raw"); ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--camera", default="front")
    a = ap.parse_args()
    raw = ROOT / a.raw; tmp = Path("/tmp") / f"clearvla_render_{a.material}_{a.camera}.png"
    files = [p for t in a.tasks.split(",") for p in sorted((raw / f"{t}_{a.material}").glob("ep_*.h5"))]
    if a.limit: files = files[:a.limit]
    # bootstrap the bridge from the first episode's scene
    with h5py.File(files[0], "r") as f: t0_task, t0_seed = f.attrs["task"], int(f.attrs["seed"])
    m0, i0 = build_scene(t0_task, a.material, t0_seed); bridge = BlenderBridge(m0, i0, a.material, samples=a.samples)
    n_frames, t_start = 0, time.perf_counter()
    for k, p in enumerate(files):
        n = render_episode(bridge, p, tmp, a.samples, a.camera); n_frames += n
        if n and (k % 10 == 0 or k == len(files) - 1):
            el = time.perf_counter() - t_start
            print(f"[{a.material}] {k+1}/{len(files)} episodes, {n_frames} frames, {el/60:.1f} min, {n_frames/max(el,1e-6):.2f} fps", flush=True)
    print(f"all rendered: {len(files)} episodes, {n_frames} new frames, {(time.perf_counter()-t_start)/60:.1f} min", flush=True)
