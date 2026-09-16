"""F6 (re-check after Week 1): wall time of one closed-loop episode per task with the scripted expert,
Blender rendering at every observation step (every 8 control steps), MuJoCo masks, and a dummy Model-A
inference on MPS. Pass condition: <= 20 s per episode.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.tasks import make_task
from experts.scripted import make_expert
from render.bridge import BlenderBridge, MaskRenderer
from scripts.bench_train import PrefixTransformer, ActionExpert

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--material", default="glass"); ap.add_argument("--engine", default="cycles")
    ap.add_argument("--samples", type=int, default=32); ap.add_argument("--obs-every", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("mps"); prefix, expert = PrefixTransformer().to(dev).eval(), ActionExpert().to(dev).eval()
    out = Path("results/week1/bench"); out.mkdir(parents=True, exist_ok=True)
    bridge = None
    for name in ("grasp", "pour", "insert"):
        task = make_task(name, a.material); obs = task.reset(a.seed); ex = make_expert(task, a.seed)
        t = {}; tic = lambda k, t0: t.__setitem__(k, t.get(k, 0.0) + time.perf_counter() - t0)
        t0 = time.perf_counter()
        if bridge is None: bridge = BlenderBridge(task.model, task.info, a.material, engine=a.engine, samples=a.samples)
        else: bridge.rebind(task.model, task.info, a.material)
        masks = MaskRenderer(task.model, task.info); tic("setup", t0)
        n_obs = 0; ep0 = time.perf_counter(); done = False
        while not done and ex.phase < len(ex.phases):
            if task.t % a.obs_every == 0:
                t0 = time.perf_counter(); bridge.sync(task.data); bridge.render(str(out / f"{name}_{a.material}_{task.t:03d}.png")); tic("blender", t0)
                t0 = time.perf_counter(); masks(task.data); tic("mask", t0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    kv = prefix(torch.randn(1, 196, 768, device=dev), torch.randn(1, 16, 768, device=dev), torch.randn(1, 14, device=dev)); x = torch.randn(1, 16, 10, device=dev)
                    for i in range(10): x = x + 0.1 * expert(x, torch.full((1,), i / 10, device=dev), kv)
                    torch.mps.synchronize()
                tic("inference", t0); n_obs += 1
            t0 = time.perf_counter(); obs, done = task.step(ex(obs)); tic("sim+expert", t0)
        ep = time.perf_counter() - ep0
        print(f"{name}/{a.material}: {task.t} steps, {n_obs} obs, success={task.success()}  TOTAL {ep:.1f} s  "
              + "  ".join(f"{k} {v:.2f}s" for k, v in t.items()) + f"  -> {'PASS' if ep <= 20 else 'FAIL'} (<=20 s)")
if __name__ == "__main__": main()
