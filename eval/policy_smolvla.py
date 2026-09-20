"""Closed-loop wrapper for a fine-tuned SmolVLA checkpoint, duck-typed to ModelAPolicy so eval/run_matrix.py
can drive it unchanged. Run in the `lerobot311` env (lerobot + mujoco + bpy).

The checkpoint dir is a lerobot `pretrained_model` folder (config.json, model.safetensors, and the dataset
normalisation stats). Observations: two 224x224 RGB views (float 0-1, CHW), 9-D joint state, the instruction
string. Actions come back denormalised as 10-D [delta-xyz, rot6d, gripper]; we add the TCP back for xyz.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, torch


class SmolVLAPolicyWrapper:
    def __init__(self, ckpt_dir: str | Path, dev=None, exec_horizon=8, n_steps=10, accel=None, ckpt_file=None):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        ckpt_dir = Path(ckpt_dir)
        if (ckpt_dir / "pretrained_model").exists(): ckpt_dir = ckpt_dir / "pretrained_model"
        self.dev = torch.device(dev or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(self.dev).eval()
        self.policy.config.n_action_steps = exec_horizon          # execute this many of the predicted chunk, then re-observe
        self.policy.config.num_steps = n_steps                    # flow-matching denoising steps
        self.views = ["front", "wrist"] if any("wrist" in k for k in self.policy.config.input_features) else ["front"]
        info = ckpt_dir.parent / "EXPORT_INFO.json"
        self.delta = True                                         # exported with delta xyz (see data/export_lerobot.py)
        self.exec_horizon, self.accel = exec_horizon, accel or {}
        self.reset()

    def reset(self):
        self.policy.reset(); self.queue = []; self.timings = []; self.diag = []; self.prev_img = None; self._n_since_obs = 0

    def need_observation(self): return self._n_since_obs % self.exec_horizon == 0

    @torch.no_grad()
    def observe(self, img_uint8, obs, task_name):
        imgs = img_uint8 if isinstance(img_uint8, (list, tuple)) else [img_uint8]
        t0 = time.perf_counter()
        batch = {"observation.state": torch.from_numpy(np.asarray(obs["qpos"], np.float32))[None].to(self.dev), "task": [obs["instruction"]]}
        for v, im in zip(self.views, imgs):
            batch[f"observation.images.{v}"] = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.dev)
        self._batch = batch; self._tcp = np.asarray(obs["tcp_pos"], np.float32).copy()
        # one call fills the policy's internal queue with a full chunk; subsequent next_action() pops from it
        a = self.policy.select_action(batch); t1 = time.perf_counter()
        self.timings.append(dict(encoder=0.0, policy=t1 - t0))
        self._pending = a; self._n_since_obs = 0
        return None

    def next_action(self):
        if self._n_since_obs == 0: a = self._pending
        else: a = self.policy.select_action(self._batch)               # pops the queue; no new inference until it empties
        a = a[0].float().cpu().numpy()[:10].copy()
        if self.delta: a[:3] += self._tcp
        a[9] = float(np.clip(a[9], 0, 1)); self._n_since_obs += 1
        return a
