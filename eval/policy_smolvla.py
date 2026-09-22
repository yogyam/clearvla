"""Closed-loop wrapper for a fine-tuned SmolVLA checkpoint, duck-typed to ModelAPolicy so eval/run_matrix.py
can drive it unchanged. Run in the `lerobot311` env (lerobot + mujoco + bpy).

The checkpoint dir is a lerobot `pretrained_model` folder (config.json, model.safetensors, and the saved
pre/post-processor pipelines). The saved preprocessor does everything the training pipeline did: renames our
`front`/`wrist` views to the pretrained `camera1`/`camera2` slots, tokenises the instruction, normalises the
9-D joint state with the dataset stats, and moves tensors to the device. The postprocessor de-normalises the
10-D action [delta-xyz, rot6d, gripper]; we add the TCP back for xyz (dataset exported with delta actions).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, torch

VIEW_KEYS = {"front": "observation.images.front", "wrist": "observation.images.wrist"}


class SmolVLAPolicyWrapper:
    def __init__(self, ckpt_dir: str | Path, dev=None, exec_horizon=8, n_steps=10, accel=None, ckpt_file=None):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        ckpt_dir = Path(ckpt_dir)
        if (ckpt_dir / "pretrained_model").exists(): ckpt_dir = ckpt_dir / "pretrained_model"
        self.dev = torch.device(dev or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(self.dev).eval()
        self.policy.config.n_action_steps = exec_horizon          # execute this many of the predicted chunk, then re-observe
        self.policy.config.num_steps = n_steps                    # flow-matching denoising steps
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": str(self.dev)}})
        rename = json.load(open(ckpt_dir / "policy_preprocessor.json"))["steps"][0].get("config", {}).get("rename_map", {})
        fed = set(rename) | set(self.policy.config.image_features)
        self.views = [v for v, k in VIEW_KEYS.items() if k in fed]   # which Blender views to render per observation
        assert self.views, f"checkpoint expects none of {list(VIEW_KEYS.values())}: {sorted(fed)}"
        self.delta = True                                         # datasets/lerobot_v2/* were exported with delta xyz (EXPORT_INFO.json)
        self.exec_horizon, self.accel = exec_horizon, accel or {}
        self.reset()

    def reset(self):
        self.policy.reset(); self.timings = []; self.diag = []; self._n_since_obs = 0

    def need_observation(self): return self._n_since_obs % self.exec_horizon == 0

    @torch.no_grad()
    def observe(self, img_uint8, obs, task_name):
        imgs = img_uint8 if isinstance(img_uint8, (list, tuple)) else [img_uint8]
        t0 = time.perf_counter()
        raw = {"observation.state": torch.from_numpy(np.asarray(obs["qpos"], np.float32)), "task": obs["instruction"]}
        for v, im in zip(self.views, imgs):
            raw[VIEW_KEYS[v]] = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        self._batch = self.pre(raw); self._tcp = np.asarray(obs["tcp_pos"], np.float32).copy()
        # one call runs inference and fills the policy's internal queue with `exec_horizon` actions
        self._pending = self._pop(); t1 = time.perf_counter()
        self.timings.append(dict(encoder=0.0, policy=t1 - t0))
        self._n_since_obs = 0
        return None

    def _pop(self):
        a = self.policy.select_action(dict(self._batch))          # pops the queue; new inference only when it empties
        return self.post(a)[0].float().cpu().numpy()[:10].copy()

    def next_action(self):
        a = self._pending if self._n_since_obs == 0 else self._pop()
        if self.delta: a[:3] += self._tcp
        a[9] = float(np.clip(a[9], 0, 1)); self._n_since_obs += 1
        return a
