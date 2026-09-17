"""Memmap-backed training dataset for Model A.

One sample = one cached frame: SigLIP visual tokens (memmap row), text tokens (instruction), proprio
(qpos 9 + TCP 10), and the 16-step action chunk starting at that step (last action repeated past the end,
with a validity mask). Actions and proprio are z-normalised with train-split statistics.
Split is by episode: the last `n_val` seeds of each task's kept list are validation (same seeds for both materials).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("grasp", "pour", "insert"); MATERIALS = ("opaque", "glass"); TASK_ID = {t: i for i, t in enumerate(TASKS)}
CHUNK = 16


def tcp10(tcp_pos, tcp_R, grip):
    return np.concatenate([tcp_pos, tcp_R[..., :, 0], tcp_R[..., :, 1], grip[..., None]], -1)


class FrameDataset(Dataset):
    def __init__(self, split="train", n_val=15, source="rgb_blender", feat_dir="datasets/features", raw_dir="datasets/raw",
                 tasks=TASKS, materials=MATERIALS, norm=None, limit=None, views=None):
        self.views = list(views) if views else [source]
        self.feat, self.raw = ROOT / feat_dir, ROOT / raw_dir
        self.text = np.load(self.feat / "text_tokens.fp16.npy"); tidx = pd.read_parquet(self.feat / "text_index.parquet")
        self.text_row = {(r.task, int(r.instruction_id)): i for i, r in tidx.iterrows()}
        self.text_len = {i: int(r.n_tokens) for i, r in tidx.iterrows()}
        man = json.loads((self.raw / "manifest.json").read_text())
        self.cells, self.samples, self.episodes = {}, [], {}
        for t in tasks:
            kept = man["kept"][t]; val_seeds = set(kept[-n_val:]); use = (lambda s: s in val_seeds) if split == "val" else (lambda s: s not in val_seeds)
            for m in materials:
                key = f"{t}_{m}_{self.views[0]}"; idx = pd.read_parquet(self.feat / f"{key}.index.parquet")
                for v in self.views:
                    vidx = pd.read_parquet(self.feat / f"{t}_{m}_{v}.index.parquet"); assert len(vidx) == len(idx), f"view index mismatch {t}/{m}/{v}"
                    self.cells[(t, m, v)] = dict(path=self.feat / f"{t}_{m}_{v}.fp16.memmap", n=len(idx), mm=None)
                for row, (seed, step) in enumerate(zip(idx.seed.values, idx.step.values)):
                    seed, step = int(seed), int(step)
                    if not use(seed): continue
                    if (t, m, seed) not in self.episodes: self.episodes[(t, m, seed)] = self._load_episode(t, m, seed)
                    self.samples.append((t, m, seed, step, row))
        if limit: self.samples = self.samples[:limit]
        self.norm = norm or self.compute_norm()

    def _load_episode(self, t, m, seed):
        with h5py.File(self.raw / f"{t}_{m}" / f"ep_{seed:04d}.h5", "r") as f:
            act = f["action"][:].astype(np.float32)
            prop = np.concatenate([f["qpos"][:], tcp10(f["tcp_pos"][:], f["tcp_R"][:], f["gripper"][:])], -1).astype(np.float32)
            return dict(act=act, prop=prop, instr=int(f.attrs["instruction_id"]))

    def compute_norm(self):
        acts = np.concatenate([e["act"] for e in self.episodes.values()]); props = np.concatenate([e["prop"] for e in self.episodes.values()])
        return dict(act_mean=acts.mean(0).tolist(), act_std=(acts.std(0) + 1e-3).tolist(), prop_mean=props.mean(0).tolist(), prop_std=(props.std(0) + 1e-3).tolist())

    def _mm(self, t, m, v):
        c = self.cells[(t, m, v)]
        if c["mm"] is None: c["mm"] = np.memmap(c["path"], dtype=np.float16, mode="r", shape=(c["n"], 196, 768))
        return c["mm"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        t, m, seed, step, row = self.samples[i]; ep = self.episodes[(t, m, seed)]; n = self.norm
        vis = torch.from_numpy(np.concatenate([np.asarray(self._mm(t, m, v)[row]) for v in self.views]))   # (196*views,768) fp16
        tr = self.text_row[(t, ep["instr"])]; txt = torch.from_numpy(self.text[tr]); L = self.text_len[tr]
        txt_mask = torch.zeros(64, dtype=torch.bool); txt_mask[:L] = True
        prop = (ep["prop"][step] - n["prop_mean"]) / n["prop_std"]
        T = len(ep["act"]); idx = np.minimum(np.arange(step, step + CHUNK), T - 1)
        act = (ep["act"][idx] - n["act_mean"]) / n["act_std"]; act_mask = (np.arange(step, step + CHUNK) < T).astype(np.float32)
        return dict(vis=vis, txt=txt, txt_mask=txt_mask, prop=torch.tensor(prop, dtype=torch.float32),
                    act=torch.tensor(act, dtype=torch.float32), act_mask=torch.tensor(act_mask), task=TASK_ID[t], material=int(m == "glass"))


def normalise_prop(prop, norm): return (np.asarray(prop, np.float32) - np.asarray(norm["prop_mean"], np.float32)) / np.asarray(norm["prop_std"], np.float32)
def denormalise_act(act, norm): return np.asarray(act, np.float32) * np.asarray(norm["act_std"], np.float32) + np.asarray(norm["act_mean"], np.float32)
