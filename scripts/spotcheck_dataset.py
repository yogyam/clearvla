"""Stage 5: dataset spot check. 20 random (episode, step) samples as a contact sheet
(Blender RGB | alpha RGB | critical-mask overlay | TCP action arrow) plus counts and sanity asserts.

  python scripts/spotcheck_dataset.py
"""
import sys, json, random
from pathlib import Path
import numpy as np, h5py, mujoco
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from envs.scene import build_scene, home
raw = ROOT / "datasets/raw"; out = ROOT / "results/week2"; out.mkdir(parents=True, exist_ok=True)
man = json.loads((raw / "manifest.json").read_text())
rng = random.Random(0)


def project(model, data, cam_id, p, res=224):
    """World point -> pixel using the MuJoCo camera (pinhole, vertical fovy)."""
    R = data.cam_xmat[cam_id].reshape(3, 3); c = data.cam_xpos[cam_id]
    v = R.T @ (np.asarray(p) - c); f = (res / 2) / np.tan(np.radians(model.cam_fovy[cam_id]) / 2)
    return res / 2 + f * v[0] / -v[2], res / 2 - f * v[1] / -v[2]


lines = ["# Week 2 dataset stats\n", "| Cell | Episodes | Frames (stride 4) | Mean steps | Blender rendered |", "|---|---|---|---|---|"]
problems = []; picks = []
for task in man["tasks"]:
    for material in ("opaque", "glass"):
        files = sorted((raw / f"{task}_{material}").glob("ep_*.h5")); n_frames = 0; steps = []; n_bl = 0
        for p in files:
            with h5py.File(p, "r") as f:
                n_frames += len(f["sample_steps"]); steps.append(int(f.attrs["n_steps"])); n_bl += int("rgb_blender" in f)
                a = f["action"][:]
                if not np.isfinite(a).all(): problems.append(f"NaN action in {p.name}")
                if np.abs(a[:, :3]).max() > 1.5 or a[:, 9].min() < 0 or a[:, 9].max() > 1: problems.append(f"action out of range in {p.name}")
        lines.append(f"| {task}/{material} | {len(files)} | {n_frames} | {np.mean(steps):.0f} | {n_bl}/{len(files)} |")
        for p in rng.sample(files, 2 if task != "pour" else 3): picks.append((task, material, p))
# paired params identical across materials
for task in man["tasks"]:
    for s in man["kept"][task][:20]:
        po = json.loads(h5py.File(raw / f"{task}_opaque" / f"ep_{s:04d}.h5", "r").attrs["params"]); pg = json.loads(h5py.File(raw / f"{task}_glass" / f"ep_{s:04d}.h5", "r").attrs["params"])
        if po != pg: problems.append(f"params differ for {task} seed {s}")
lines.append(f"\nSanity problems: {len(problems)} {problems[:5]}")

S = 224; rows = []
for task, material, p in picks[:20]:
    with h5py.File(p, "r") as f:
        i = rng.randrange(len(f["sample_steps"])); t = int(f["sample_steps"][i])
        bl = f["rgb_blender"][i] if "rgb_blender" in f else np.zeros((S, S, 3), np.uint8)
        al = f["rgb_alpha"][i]; mk = f["mask_crit"][i]; tcp = f["tcp_pos"][t]; act = f["action"][min(t + 8, len(f["action"]) - 1)]
        seed = int(f.attrs["seed"]); instr = f.attrs["instruction"]
    ov = al.copy().astype(float); ov[mk] = 0.4 * ov[mk] + 0.6 * np.array([230, 40, 40]); ov = ov.astype(np.uint8)
    model, info = build_scene(task, material, seed); data = mujoco.MjData(model); home(model, data); mujoco.mj_forward(model, data)
    x0, y0 = project(model, data, info.cam_id, tcp); x1, y1 = project(model, data, info.cam_id, act[:3])
    arrow = Image.fromarray(bl.copy()); d = ImageDraw.Draw(arrow); d.line([(x0, y0), (x1, y1)], fill=(255, 220, 0), width=2); d.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], fill=(255, 220, 0))
    d.text((4, 4), f"{task}/{material} s{seed} t{t}", fill=(255, 255, 0)); d.text((4, S - 14), instr[:36], fill=(255, 255, 0))
    rows.append(np.concatenate([np.asarray(arrow), al, ov], 1))
sheet = np.concatenate(rows, 0); Image.fromarray(sheet).save(out / "spotcheck.png")
lines.append(f"\nSpot check sheet: results/week2/spotcheck.png ({len(rows)} samples; columns = Blender + action arrow (8 steps ahead) | MuJoCo-alpha | critical mask overlay)")
(out / "dataset_stats.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
