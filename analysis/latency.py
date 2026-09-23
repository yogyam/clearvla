"""Controlled latency measurement on the Mac (MPS, eager PyTorch): median ms per observation over N runs, split into
SigLIP (two views) / prefix / 10 denoising steps, for each accel config. Run when nothing else uses the GPU.
  python analysis/latency.py --n 300 -> results/week7/latency_mps.json
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from eval.policy import ModelAPolicy
from accel import make_accel


def sync(dev):
    if dev.type == "mps": torch.mps.synchronize()
    elif dev.type == "cuda": torch.cuda.synchronize()


def measure(name, n, dev=None):
    pol = ModelAPolicy(ROOT / "checkpoints/v4", accel=make_accel(name)); pol.reset()
    imgs = [np.random.randint(0, 255, (224, 224, 3), np.uint8) for _ in pol.views]
    obs = dict(qpos=np.zeros(9, np.float32), tcp_pos=np.zeros(3, np.float32), tcp_R=np.eye(3, dtype=np.float32), gripper=1.0, instruction_id=0, instruction="")
    enc, pre, exp = [], [], []
    with torch.no_grad():
        for it in range(n + 20):
            if pol.accel is not None and pol.accel.needs_gt: pol.accel.gt_crit = np.zeros(pol.n_vis, bool)
            t0 = time.perf_counter(); vis = torch.cat([pol.enc(im) for im in imgs], 1); sync(pol.dev); t1 = time.perf_counter()
            row, L = pol.text_row[("grasp", 0)]; txt = torch.from_numpy(pol.text[row])[None].to(pol.dev); txt_mask = torch.zeros(1, 64, dtype=torch.bool, device=pol.dev); txt_mask[0, :L] = True
            prop_t = torch.zeros(1, 19, device=pol.dev)
            kw = pol.accel.act_kwargs(pol.n_vis, imgs, txt_mask, pol.dev, vis=vis) if pol.accel is not None else {}
            with torch.autocast(device_type=pol.dev.type, dtype=torch.float16):
                mem, mask, index, _ = pol.model.prefix(vis, txt, txt_mask, prop_t, **kw); sync(pol.dev); t2 = time.perf_counter()
                x = torch.randn(1, 16, 10, device=pol.dev); ctx = None
                for i in range(10):
                    t = torch.full((1,), i / 10, device=pol.dev); v, kvs = pol.model.expert(x, t, mem, mask, ctx_kv=ctx); ctx = kvs; x = x + v / 10
                sync(pol.dev); t3 = time.perf_counter()
            # keep the cache's "previous frame" changing so the cache path is exercised (not the first-obs full path)
            imgs = [np.clip(im.astype(int) + np.random.randint(-3, 4, im.shape), 0, 255).astype(np.uint8) for im in imgs]
            if it >= 20: enc.append(t1 - t0); pre.append(t2 - t1); exp.append(t3 - t2)
    ms = lambda a: float(np.median(a) * 1000)
    return dict(config=name, siglip_ms=ms(enc), prefix_ms=ms(pre), expert_ms=ms(exp), total_ms=ms(np.array(enc) + np.array(pre) + np.array(exp)), n=n, tokens_after_l2=int(index.shape[1]),
                gflops=pol.accel.last_cost["total_gflops"] if pol.accel is not None else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--configs", default="full,prune50,prune25,prune12,cache50,cache25,quant4"); a = ap.parse_args()
    out = []; (ROOT / "results/week7").mkdir(exist_ok=True, parents=True)
    for c in a.configs.split(","):
        r = measure(c, a.n); out.append(r); print(json.dumps(r))
    json.dump(dict(device="mps", torch=torch.__version__, runs=out), open(ROOT / "results/week7/latency_mps.json", "w"), indent=1)
