"""F6: wall time of one closed-loop episode with a random policy and Blender rendering.

150 control steps at 10 Hz, observe every 8 steps (~19 observations). Per observation:
Blender RGB (Cycles/Metal), MuJoCo segmentation mask, and a dummy Model-A inference
(prefix once + 10 Euler steps) on MPS. Pass condition: <= 20 s per episode.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, mujoco, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render.bridge import BlenderBridge, MaskRenderer
from scripts.bench_train import PrefixTransformer, ActionExpert

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="glass"); ap.add_argument("--engine", default="cycles")
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--obs-every", type=int, default=8)
    ap.add_argument("--samples", type=int, default=32); ap.add_argument("--out", default="results/week0/episode")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(root / "envs/lab_scene_v0.xml")); data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0: mujoco.mj_resetDataKeyframe(model, data, key)
    # keyframe "home" covers only the arm; place the tube free joint explicitly
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tube_free"); adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr+7] = [0.5, 0.0, 0.051, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    t = {}
    def tic(k, t0): t[k] = t.get(k, 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    bridge = BlenderBridge(model, a.material, root / "envs/franka/assets", engine=a.engine, samples=a.samples)
    masks = MaskRenderer(model)
    tic("setup(once)", t0); print(f"robot meshes imported into Blender: {bridge.n_robot_meshes}")
    dev = torch.device("mps"); prefix, expert = PrefixTransformer().to(dev).eval(), ActionExpert().to(dev).eval()
    rng = np.random.default_rng(0); home = data.ctrl.copy(); substeps = int(round(0.1 / model.opt.timestep))
    n_obs = 0
    ep0 = time.perf_counter()
    for step in range(a.steps):
        if step % a.obs_every == 0:
            t0 = time.perf_counter(); bridge.sync(data); bridge.render(str(out / f"{a.material}_{a.engine}_{step:03d}.png")); tic("blender", t0)
            t0 = time.perf_counter(); m = masks(data); tic("mask", t0)
            t0 = time.perf_counter()
            with torch.no_grad():
                vis = torch.randn(1,196,768, device=dev); txt = torch.randn(1,16,768, device=dev); prop = torch.randn(1,14, device=dev)
                kv = prefix(vis, txt, prop); x = torch.randn(1,16,10, device=dev)
                for i in range(10): x = x + 0.1 * expert(x, torch.full((1,), i/10, device=dev), kv)
                torch.mps.synchronize()
            tic("inference", t0); n_obs += 1
            if step == 0: print(f"mask pixels on tube: {int(m.sum())}")
        t0 = time.perf_counter()
        data.ctrl[:] = home + rng.normal(0, 0.05, size=home.shape)
        for _ in range(substeps): mujoco.mj_step(model, data)
        tic("sim", t0)
    ep = time.perf_counter() - ep0
    print(f"episode: {a.steps} steps, {n_obs} observations, material={a.material}, engine={a.engine}, samples={a.samples}")
    for k, v in t.items(): print(f"  {k:12s} {v:7.2f} s" + (f"  ({v/n_obs*1000:.0f} ms/obs)" if k in ('blender','mask','inference') else ""))
    print(f"  {'TOTAL':12s} {ep:7.2f} s  -> {'F6 PASS' if ep <= 20 else 'F6 FAIL'} (<= 20 s)")
if __name__ == "__main__": main()
