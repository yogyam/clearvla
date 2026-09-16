"""Stage 4: export recorded episodes to LeRobot format (one dataset per task, both materials inside).

Runs in the `lerobot` conda env. Frames are written at the 10 Hz control rate; the camera image is the
rendered frame at the last sampled step (held for `stride` steps) and `frame_is_new` marks the steps that
have a fresh render, so a dataloader can subsample to those if it wants only true observations.

  conda activate lerobot && python data/export_lerobot.py --source rgb_blender
"""
import argparse, sys, json, shutil
from pathlib import Path
import numpy as np, h5py
ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ("opaque", "glass"); MAT_ID = {"opaque": 0.0, "glass": 1.0}
FPS = 10


def features(res=224):
    return {
        "observation.images.front": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (9,), "names": [f"joint{i}" for i in range(1, 8)] + ["finger1", "finger2"]},
        "observation.tcp": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"]},
        "action": {"dtype": "float32", "shape": (10,), "names": ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"]},
        "material": {"dtype": "float32", "shape": (1,), "names": ["material_id"]},
        "seed": {"dtype": "float32", "shape": (1,), "names": ["seed"]},
        "frame_is_new": {"dtype": "float32", "shape": (1,), "names": ["is_new"]},
        "fill_level": {"dtype": "float32", "shape": (1,), "names": ["receiver_fill"]},
    }


def export_task(task, raw, out_root, source, limit):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = out_root / f"clearvla_{task}"
    if root.exists(): shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=f"clearvla/{task}", fps=FPS, features=features(), root=root, robot_type="franka_panda", use_videos=True,
                               image_writer_threads=8, vcodec="h264")
    n_ep = 0
    for material in MATERIALS:
        files = sorted((raw / f"{task}_{material}").glob("ep_*.h5"))[: limit or None]
        for p in files:
            with h5py.File(p, "r") as f:
                steps = f["sample_steps"][:]; imgs = f[source][:]; T = int(f.attrs["n_steps"])
                qpos, act, tcp_p, tcp_R, grip, fill = f["qpos"][:], f["action"][:], f["tcp_pos"][:], f["tcp_R"][:], f["gripper"][:], f["fill"][:]
                seed = int(f.attrs["seed"]); instr = str(f.attrs["instruction"])
            j = -1
            for t in range(T):
                is_new = t in steps
                if is_new: j += 1
                tcp10 = np.concatenate([tcp_p[t], tcp_R[t][:, 0], tcp_R[t][:, 1], [grip[t]]]).astype(np.float32)
                ds.add_frame({"observation.images.front": imgs[j], "observation.state": qpos[t].astype(np.float32), "observation.tcp": tcp10,
                              "action": act[t].astype(np.float32), "material": np.array([MAT_ID[material]], np.float32),
                              "seed": np.array([seed], np.float32), "frame_is_new": np.array([float(is_new)], np.float32),
                              "fill_level": np.array([fill[t]], np.float32), "task": instr})
            ds.save_episode(); n_ep += 1
    print(f"{task}: {n_ep} episodes, {ds.meta.total_frames} frames -> {root}", flush=True)
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="grasp,pour,insert"); ap.add_argument("--raw", default="datasets/raw")
    ap.add_argument("--out", default="datasets/lerobot"); ap.add_argument("--source", default="rgb_blender", choices=["rgb_blender", "rgb_alpha"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    for t in a.tasks.split(","): export_task(t, ROOT / a.raw, out, a.source, a.limit)
    (out / "EXPORT_INFO.json").write_text(json.dumps(dict(source=a.source, fps=FPS, stride_note="frame_is_new marks steps with a fresh render", limit=a.limit), indent=1))
