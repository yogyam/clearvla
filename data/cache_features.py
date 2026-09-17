"""Stage 3: SigLIP ViT-B/16 patch-feature cache for every rendered frame.

One fp16 memmap per task/material: (N, 196, 768). An index parquet maps row -> (seed, step). Text-side
tokens for the 24 instructions are cached once. Training reads rows from disk; nothing is held in RAM.

  python data/cache_features.py --source rgb_blender
  python data/cache_features.py --verify
"""
import argparse, sys, json, time
from pathlib import Path
import numpy as np, h5py, torch, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
MODEL = "google/siglip-base-patch16-224"
MATERIALS = ("opaque", "glass")


def load_model(dev):
    from transformers import AutoModel, AutoProcessor
    m = AutoModel.from_pretrained(MODEL).to(dev).eval(); p = AutoProcessor.from_pretrained(MODEL)
    return m, p


@torch.no_grad()
def vision_tokens(model, proc, imgs_uint8, dev):
    x = proc(images=list(imgs_uint8), return_tensors="pt")["pixel_values"].to(dev)
    return model.vision_model(pixel_values=x).last_hidden_state.to(torch.float16).cpu().numpy()


def cache_cell(task, material, raw, feat, source, model, proc, dev, batch):
    files = sorted((raw / f"{task}_{material}").glob("ep_*.h5"))
    index = []
    for p in files:
        with h5py.File(p, "r") as f: index += [(int(f.attrs["seed"]), int(s)) for s in f["sample_steps"][:]]
    N = len(index); out = feat / f"{task}_{material}_{source}.fp16.memmap"
    mm = np.memmap(out, dtype=np.float16, mode="w+", shape=(N, 196, 768))
    row, t0 = 0, time.perf_counter()
    for p in files:
        with h5py.File(p, "r") as f: imgs = f[source][:]
        for i in range(0, len(imgs), batch):
            feats = vision_tokens(model, proc, imgs[i:i + batch], dev); mm[row:row + len(feats)] = feats; row += len(feats)
    mm.flush(); del mm
    pd.DataFrame(index, columns=["seed", "step"]).to_parquet(feat / f"{task}_{material}_{source}.index.parquet")
    print(f"{task}/{material}/{source}: {N} frames, {N*196*768*2/2**30:.2f} GB, {N/(time.perf_counter()-t0):.0f} fps", flush=True)
    return N


def cache_text(model, proc, feat, dev):
    import yaml
    instr = yaml.safe_load((ROOT / "envs/instructions.yaml").read_text())
    rows, toks = [], []
    with torch.no_grad():
        for task, lst in instr.items():
            for i, s in enumerate(lst):
                t = proc.tokenizer(s, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
                h = model.text_model(**t).last_hidden_state[0].to(torch.float16).cpu().numpy()
                rows.append(dict(task=task, instruction_id=i, text=s, n_tokens=int(t["attention_mask"].sum()) if "attention_mask" in t else 64)); toks.append(h)
    np.save(feat / "text_tokens.fp16.npy", np.stack(toks)); pd.DataFrame(rows).to_parquet(feat / "text_index.parquet")
    print(f"text: {len(rows)} instructions x {toks[0].shape}")


def verify(raw, feat, source, model, proc, dev, n=100):
    rng = np.random.default_rng(0); worst = 0.0
    for task in ("grasp", "pour", "insert"):
        for material in MATERIALS:
            idx = pd.read_parquet(feat / f"{task}_{material}_{source}.index.parquet")
            mm = np.memmap(feat / f"{task}_{material}_{source}.fp16.memmap", dtype=np.float16, mode="r", shape=(len(idx), 196, 768))
            for r in rng.choice(len(idx), size=min(n // 6, len(idx)), replace=False):
                seed, step = idx.iloc[r]
                with h5py.File(raw / f"{task}_{material}" / f"ep_{seed:04d}.h5", "r") as f:
                    i = list(f["sample_steps"][:]).index(step); img = f[source][i]
                ref = vision_tokens(model, proc, img[None], dev)[0].astype(np.float32)
                err = np.abs(ref - mm[r].astype(np.float32)).max(); worst = max(worst, err)
    print(f"verify: max abs error over {n} rows = {worst:.4f} ({'PASS' if worst < 0.05 else 'FAIL'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw"); ap.add_argument("--feat", default="datasets/features")
    ap.add_argument("--source", default="rgb_blender"); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tasks", default="grasp,pour,insert"); ap.add_argument("--materials", default="opaque,glass"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(); raw, feat = ROOT / a.raw, ROOT / a.feat; feat.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; model, proc = load_model(dev)
    if a.verify: verify(raw, feat, a.source, model, proc, dev); sys.exit()
    total = sum(cache_cell(t, m, raw, feat, a.source, model, proc, dev, a.batch) for t in a.tasks.split(",") for m in a.materials.split(","))
    if "opaque" in a.materials: cache_text(model, proc, feat, dev)
    print(f"cached {total} frames from {a.source}")
