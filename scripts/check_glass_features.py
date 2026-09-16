"""F4: does glass look hard to the frozen encoder?

Loads paired glass/opaque renders (same scene, same camera) produced by bench_render.py,
runs SigLIP ViT-B/16, and reports per-patch cosine distance between materials, split into
patches ON the tube vs OFF the tube. On-tube distance >> off-tube distance means the
material change is visible to the encoder and localised to the object.
Also writes a side-by-side PNG and a patch-diff heatmap for the eyeball check.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip-base-patch16-224"

def patch_feats(model, proc, img, dev):
    x = proc(images=img, return_tensors="pt")["pixel_values"].to(dev)
    with torch.no_grad():
        out = model.vision_model(pixel_values=x)
    return out.last_hidden_state[0].float().cpu()  # (196, 768)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="cycles"); ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--dir", default="results/week0/render")
    # tube bbox in image coords (x0,y0,x1,y1) in 224 px; default from the bench scene camera. Refine with a mask later.
    ap.add_argument("--tube-box", default="auto")
    a = ap.parse_args()
    d = Path(a.dir)
    g = Image.open(d / f"{a.engine}_glass_{a.frame:03d}.png").convert("RGB")
    o = Image.open(d / f"{a.engine}_opaque_{a.frame:03d}.png").convert("RGB")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL).to(dev).eval(); proc = AutoProcessor.from_pretrained(MODEL)
    fg, fo = patch_feats(model, proc, g, dev), patch_feats(model, proc, o, dev)
    cos = torch.nn.functional.cosine_similarity(fg, fo, dim=-1)            # (196,)
    dist = (1 - cos).reshape(14, 14).numpy()
    if a.tube_box == "auto":
        # patches whose pixels differ between materials: the tube is the only thing that changed
        diff = np.abs(np.asarray(g).astype(float) - np.asarray(o).astype(float)).mean(-1)
        ys, xs = np.where(diff > 12)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
        print(f"  auto tube box from pixel diff: {x0},{y0},{x1},{y1}")
    else:
        x0, y0, x1, y1 = map(int, a.tube_box.split(","))
    on = np.zeros((14, 14), bool); on[y0//16:(y1+15)//16, x0//16:(x1+15)//16] = True
    print(f"SigLIP patch cosine DISTANCE glass vs opaque ({a.engine}, frame {a.frame})")
    print(f"  on-tube patches  (n={on.sum():3d}): mean {dist[on].mean():.3f}  max {dist[on].max():.3f}")
    print(f"  off-tube patches (n={(~on).sum():3d}): mean {dist[~on].mean():.3f}  max {dist[~on].max():.3f}")
    ratio = dist[on].mean() / max(dist[~on].mean(), 1e-6)
    print(f"  on/off ratio: {ratio:.2f}  -> {'PASS' if ratio > 2 else 'WEAK'} (want localised, >2x)")
    # pixel-level: how faint is the tube in the glass image vs its background?
    ga = np.asarray(g).astype(float); oa = np.asarray(o).astype(float)
    tube_px = np.zeros((224,224), bool); tube_px[y0:y1, x0:x1] = True
    for name, arr in (("glass", ga), ("opaque", oa)):
        inside, outside = arr[tube_px].mean(0), arr[~tube_px].mean(0)
        print(f"  {name:6s} mean RGB on-tube {inside.round(1)}  off-tube {outside.round(1)}  |diff| {np.abs(inside-outside).mean():.1f}")
    # visuals
    side = Image.new("RGB", (224*2+8, 224), "white"); side.paste(o, (0,0)); side.paste(g, (232,0))
    side.save(d / f"f4_side_by_side_{a.engine}.png")
    hm = (dist / dist.max() * 255).astype(np.uint8)
    Image.fromarray(hm).resize((224,224), Image.NEAREST).save(d / f"f4_patch_dist_{a.engine}.png")
    print(f"  wrote {d}/f4_side_by_side_{a.engine}.png and f4_patch_dist_{a.engine}.png")
if __name__ == "__main__": main()
