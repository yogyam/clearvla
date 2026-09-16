"""Tile review frames into one PNG per task: rows = (seed, material), columns = time."""
import sys, re, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/week1/frames"); out = d.parent
groups = collections.defaultdict(list)
for f in sorted(d.glob("*.png")):
    m = re.match(r"(\w+?)_(opaque|glass|alpha)_s(\d+)_t(\d+)\.png", f.name)
    if m: groups[(m[1], m[2], int(m[3]))].append((int(m[4]), f))
for task in sorted({k[0] for k in groups}):
    rows = [k for k in sorted(groups) if k[0] == task]
    ncol = max(len(groups[k]) for k in rows); S = 224; pad = 4; label_w = 110
    sheet = Image.new("RGB", (label_w + ncol * (S + pad), len(rows) * (S + pad)), "white"); dr = ImageDraw.Draw(sheet)
    for r, k in enumerate(rows):
        dr.text((6, r * (S + pad) + S // 2 - 6), f"{k[1]} seed {k[2]}", fill="black")
        for c, (t, f) in enumerate(sorted(groups[k])):
            sheet.paste(Image.open(f).convert("RGB"), (label_w + c * (S + pad), r * (S + pad)))
            dr.text((label_w + c * (S + pad) + 4, r * (S + pad) + 4), f"t{t}", fill="yellow")
    sheet.save(out / f"review_{task}.png"); print("wrote", out / f"review_{task}.png", sheet.size)
