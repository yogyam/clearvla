"""Closed-loop policy wrapper for Model A: live SigLIP on the rendered frame -> prefix -> 10 Euler steps ->
16 actions, execute the first `exec_horizon`, re-observe. Acceleration configs plug in via `accel` (Week 4).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, torch
from model.vla import ClearVLA, CHUNK
from data.dataset import normalise_prop, denormalise_act, tcp10

SIGLIP = "google/siglip-base-patch16-224"


class LiveSigLIP:
    def __init__(self, dev):
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained(SIGLIP).to(dev).eval(); self.p = AutoProcessor.from_pretrained(SIGLIP); self.dev = dev
    @torch.no_grad()
    def __call__(self, img_uint8):
        x = self.p(images=[img_uint8], return_tensors="pt")["pixel_values"].to(self.dev)
        return self.m.vision_model(pixel_values=x).last_hidden_state.to(torch.float16)      # (1,196,768)


class ModelAPolicy:
    def __init__(self, ckpt_dir: str | Path, dev=None, exec_horizon=8, n_steps=10, accel=None, ckpt_file="best.pt"):
        ckpt_dir = Path(ckpt_dir); self.dev = torch.device(dev or ("mps" if torch.backends.mps.is_available() else "cpu"))
        state = torch.load(ckpt_dir / ckpt_file, map_location="cpu")   # best.pt (EMA, best val) or last.pt (EMA at the final step)
        n_vis = state["model"]["prefix.pos_vis"].shape[1]; self.views = ["front", "wrist"][: n_vis // 196]
        self.model = ClearVLA(n_vis=n_vis).to(self.dev).eval(); self.model.load_state_dict(state["model"])
        self.norm = json.load(open(ckpt_dir / "norm.json")); self.enc = LiveSigLIP(self.dev)
        feat = ckpt_dir.parents[1] / "datasets/features"
        import pandas as pd
        self.text = np.load(feat / "text_tokens.fp16.npy"); tidx = pd.read_parquet(feat / "text_index.parquet")
        self.text_row = {(r.task, int(r.instruction_id)): (i, int(r.n_tokens)) for i, r in tidx.iterrows()}
        self.exec_horizon, self.n_steps, self.accel = exec_horizon, n_steps, accel or {}
        self.reset()

    def reset(self):
        self.queue = []; self.timings = []; self.diag = []; self.prev_img = None; self.step_in_chunk = 0

    def need_observation(self): return len(self.queue) == 0

    @torch.no_grad()
    def observe(self, img_uint8, obs, task_name):
        """Consume rendered frame(s) (one per view) + low-dim obs, refill the action queue with `exec_horizon` actions."""
        t0 = time.perf_counter()
        imgs = img_uint8 if isinstance(img_uint8, (list, tuple)) else [img_uint8]
        vis = torch.cat([self.enc(im) for im in imgs], 1); t1 = time.perf_counter()
        row, L = self.text_row[(task_name, obs["instruction_id"])]
        txt = torch.from_numpy(self.text[row])[None].to(self.dev); txt_mask = torch.zeros(1, 64, dtype=torch.bool, device=self.dev); txt_mask[0, :L] = True
        prop = np.concatenate([obs["qpos"], tcp10(obs["tcp_pos"], obs["tcp_R"], np.float32(obs["gripper"]))]).astype(np.float32)
        prop_t = torch.from_numpy(normalise_prop(prop, self.norm))[None].to(self.dev)
        keep_visual = self.accel.get("keep_fn", lambda *_: None)(self, vis, img_uint8)
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16):
            chunk, d = self.model.act(vis, txt, txt_mask, prop_t, n_steps=self.n_steps, keep_visual=keep_visual, return_attn=self.accel.get("return_attn", False))
        t2 = time.perf_counter()
        acts = denormalise_act(chunk[0].float().cpu().numpy(), self.norm)
        if self.norm.get("action_mode", "abs") == "delta": acts[:, :3] += np.asarray(obs["tcp_pos"], np.float32)
        self.queue = list(acts[: self.exec_horizon]); self.prev_img = img_uint8
        self.timings.append(dict(encoder=t1 - t0, policy=t2 - t1))
        if keep_visual is not None: self.diag.append(keep_visual[0].cpu().numpy().astype(bool))
        return acts

    def next_action(self, obs=None):
        a = self.queue.pop(0); a[9] = float(np.clip(a[9], 0, 1)); return a
