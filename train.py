"""Train Model A on the feature cache.

  python train.py --config configs/model_a.yaml --run smoke --steps 5000
  python train.py --config configs/model_a.yaml --run full
  python train.py --overfit 64 --steps 500 --run overfit
"""
import argparse, json, math, sys, time, copy
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model.vla import ClearVLA
from data.dataset import FrameDataset

DEFAULT = dict(views=["rgb_blender"], action_mode="abs", lr=1e-4, lr_min=1e-5, warmup=1000, wd=0.01, batch=64, steps=50000, clip=1.0, ema=0.999, amp="fp16",
               val_every=1000, val_batches=8, ckpt_every=5000, n_val=15, seed=0)


def lr_at(step, cfg):
    if step < cfg["warmup"]: return cfg["lr"] * step / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * p))


def to_dev(b, dev): return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_a.yaml"); ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int); ap.add_argument("--overfit", type=int, default=0); ap.add_argument("--amp", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/last.pt")
    a = ap.parse_args()
    cfg = dict(DEFAULT); cfg.update(yaml.safe_load(open(ROOT / a.config)) if (ROOT / a.config).exists() else {})
    if a.steps: cfg["steps"] = a.steps
    if a.amp: cfg["amp"] = a.amp
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = ROOT / "checkpoints" / a.run; out.mkdir(parents=True, exist_ok=True)
    res = ROOT / "results/week3"; res.mkdir(parents=True, exist_ok=True)

    train = FrameDataset("train", n_val=cfg["n_val"], limit=a.overfit or None, views=cfg["views"], action_mode=cfg["action_mode"])
    val = FrameDataset("val", n_val=cfg["n_val"], norm=train.norm, views=cfg["views"], action_mode=cfg["action_mode"]) if not a.overfit else train
    json.dump(train.norm, open(out / "norm.json", "w")); json.dump(cfg, open(out / "config.json", "w"))
    print(f"train {len(train)} frames / {len(train.episodes)} episodes; val {len(val)} frames", flush=True)
    dl = DataLoader(train, batch_size=cfg["batch"], shuffle=True, drop_last=True, num_workers=0)
    vdl = DataLoader(val, batch_size=cfg["batch"], shuffle=True, drop_last=False, num_workers=0)

    model = ClearVLA(n_vis=196 * len(cfg["views"])).to(dev); ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    print(f"params: {model.n_params()/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.95))
    amp = cfg["amp"] != "none"; dtype = torch.float16 if cfg["amp"] == "fp16" else torch.bfloat16
    log = open(res / f"train_log_{a.run}.jsonl", "a"); step, t0, best = 0, time.perf_counter(), float("inf")
    losses = []; t_run = time.perf_counter()
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu")
        model.load_state_dict(ck["raw"]); ema.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = int(ck["step"])
        if (out / "best.pt").exists(): best = float(torch.load(out / "best.pt", map_location="cpu")["val"])
        t0 = time.perf_counter() - step / 1.0   # keeps the it/s print sane
        print(f"resumed from step {step} (best val so far {best:.4f})", flush=True)
    t_run = time.perf_counter()
    while step < cfg["steps"]:
        for batch in dl:
            if step >= cfg["steps"]: break
            for g in opt.param_groups: g["lr"] = lr_at(step, cfg)
            batch = to_dev(batch, dev)
            with torch.autocast(device_type=dev.type, dtype=dtype, enabled=amp):
                loss = model.loss(batch)
            if not torch.isfinite(loss): print(f"non-finite loss at step {step}; stopping", flush=True); sys.exit(2)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(cfg["ema"]).add_(pm.detach(), alpha=1 - cfg["ema"])
            losses.append(loss.item()); step += 1
            if step % 100 == 0:
                rec = dict(step=step, loss=float(np.mean(losses[-100:])), lr=lr_at(step, cfg), gn=float(gn), it_s=len(losses) / (time.perf_counter() - t_run))
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if step % cfg["val_every"] == 0 or step == cfg["steps"]:
                vl = []
                with torch.no_grad():
                    for i, vb in enumerate(vdl):
                        if i >= cfg["val_batches"]: break
                        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=amp): vl.append(ema.loss(to_dev(vb, dev)).item())
                v = float(np.mean(vl)); log.write(json.dumps(dict(step=step, val_loss=v)) + "\n"); log.flush(); print(f"  val {v:.4f}", flush=True)
                if v < best: best = v; torch.save(dict(model=ema.state_dict(), step=step, val=v, cfg=cfg), out / "best.pt")
            if step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
                torch.save(dict(model=ema.state_dict(), raw=model.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg), out / "last.pt")
    print(f"done {step} steps in {(time.perf_counter()-t0)/60:.1f} min; best val {best:.4f}", flush=True)


if __name__ == "__main__": main()
