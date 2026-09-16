"""Closed-loop episode runner shared by expert verification (Week 1), demo recording (Week 2) and policy
evaluation (Week 5). Rendering is pluggable: None (state only), "mujoco" (alpha tier), or a BlenderBridge.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, time
from pathlib import Path
import numpy as np
import mujoco

from envs.tasks import make_task, HORIZON
from envs.scene import TUBE_H, TUBE_WALL, BEAKER_H


@dataclass
class EpisodeLog:
    task: str; material: str; seed: int
    success: bool = False; steps: int = 0; failure: str = ""
    instruction_id: int = -1; instruction: str = ""
    fill_trace: list = field(default_factory=list)
    wall_s: float = 0.0
    def to_json(self): return json.dumps(asdict(self))


class MujocoFrames:
    """RGB + task-critical mask from the MuJoCo renderer (dev tier / masks)."""
    def __init__(self, model, info, res=224, camera="front"):
        self.r = mujoco.Renderer(model, height=res, width=res); self.cam = camera; self.crit = np.array(info.critical_geoms)
        self.tube = info.tube_geom
    def rgb(self, data):
        self.r.disable_segmentation_rendering(); self.r.update_scene(data, camera=self.cam); return self.r.render().copy()
    def masks(self, data):
        self.r.enable_segmentation_rendering(); self.r.update_scene(data, camera=self.cam); seg = self.r.render()
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        return (np.isin(seg[..., 0], self.crit) & is_geom), ((seg[..., 0] == self.tube) & is_geom)


def run_episode(task, policy, seed: int, render: str | None = None, bridge=None, frames_dir: Path | None = None,
                frame_every: int = 8, on_step=None, max_steps: int = HORIZON, early_stop: bool = True) -> EpisodeLog:
    t0 = time.perf_counter()
    obs = task.reset(seed)
    log = EpisodeLog(task.name, task.material, seed, instruction_id=obs["instruction_id"], instruction=obs["instruction"])
    mj = MujocoFrames(task.model, task.info) if (render or frames_dir) else None
    if bridge is not None: bridge.rebind(task.model, task.info)
    done = False; idle = 0
    while not done:
        if (render or frames_dir) and task.t % frame_every == 0:
            if bridge is not None:
                bridge.sync(task.data, fill=task.fill_level(), source_fill=(task.liquid.source if task.liquid else None))
                if frames_dir: bridge.render(str(frames_dir / f"{task.name}_{task.material}_s{seed}_t{task.t:03d}.png"))
            elif frames_dir:
                from PIL import Image; Image.fromarray(mj.rgb(task.data)).save(frames_dir / f"{task.name}_{task.material}_s{seed}_t{task.t:03d}.png")
        if on_step: on_step(task, obs, mj)
        action = policy(obs)
        obs, done = task.step(action)
        if early_stop and getattr(policy, "phase", 0) >= len(getattr(policy, "phases", [])):
            idle += 1
            if idle >= 5: break
    log.success = task.success(); log.steps = task.t
    log.failure = "" if log.success else task.failure_reason()
    if task.liquid: log.fill_trace = [round(x, 4) for x in task.liquid.trace]
    log.wall_s = time.perf_counter() - t0
    return log
