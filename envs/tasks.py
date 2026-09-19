"""The three tasks. Each owns a model/data pair, a controller, success and failure checks, and instruction sampling.

Task.reset(seed) -> obs ; Task.step(action) -> (obs, done) ; Task.success() -> bool ; Task.failure_reason() -> str
obs is a dict of low-dimensional state; images are produced by the runner (MuJoCo or Blender) so the task
stays renderer-agnostic.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import mujoco
import yaml

from envs.scene import (build_scene, home, reset_tube_pose, set_liquid_height, SceneInfo,
                        TUBE_H, TUBE_WALL, TUBE_R, BEAKER_H, HOLDER_SLOT, RACK_SLOT)
from envs.control import Controller, ARM_DOF
from envs.liquid import LiquidState, tube_tilt, opening_world, pour_point_world, over_receiver

INSTRUCTIONS = yaml.safe_load((Path(__file__).parent / "instructions.yaml").read_text())
HORIZON = 300
INSERT_HOLD = 0.035        # tube centre sits this far below the TCP when it starts in-hand (fingers clear the holder walls)
SETTLE_STEPS = 100          # physics steps after reset
# Sticky gripper: when the gripper is commanded closed and the tube is between the pads, a weld constraint
# attaches it to the hand; commanding open releases it. Contact physics stays on. This removes grasp-slip
# noise (a contact-model artefact of the Menagerie pads) from what is a perception study. Reported in the paper.
PINCH_LATERAL = 0.020       # tube axis within this distance of the TCP in the pad plane


class Task:
    name = "base"

    def __init__(self, material: str = "opaque", randomize: bool = True):
        self.material, self.randomize = material, randomize
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.info: SceneInfo | None = None
        self.ctrl: Controller | None = None
        self.t = 0
        self.instruction = ""; self.instruction_id = -1
        self.liquid: LiquidState | None = None
        self.attached = False

    # ---- lifecycle -------------------------------------------------------------------------
    def reset(self, seed: int):
        self.seed = seed
        self.model, self.info = build_scene(self.name, self.material, seed, self.randomize)
        self.data = mujoco.MjData(self.model)
        home(self.model, self.data)
        self.ctrl = Controller(self.model, self.info.tcp_site)
        rng = np.random.default_rng(seed + 10_000)
        self.instruction_id = int(rng.integers(len(INSTRUCTIONS[self.name])))
        self.instruction = INSTRUCTIONS[self.name][self.instruction_id]
        self.t = 0
        self.attached = False
        self._task_reset(rng)
        for _ in range(SETTLE_STEPS): mujoco.mj_step(self.model, self.data)
        self._after_settle()
        return self.obs()

    def _task_reset(self, rng): pass
    def _after_settle(self): pass

    def step(self, action: np.ndarray):
        self.ctrl.apply_action(self.data, action)
        for _ in range(self.ctrl.substeps):
            self._update_weld(float(action[9]))
            mujoco.mj_step(self.model, self.data)
        self._task_step()
        self.t += 1
        return self.obs(), self.t >= HORIZON

    def _task_step(self): pass

    # ---- sticky gripper --------------------------------------------------------------------
    def _tube_in_pinch(self) -> bool:
        p, R = self.ctrl.tcp_pose(self.data)
        rel = R.T @ (self.data.xpos[self.info.tube_body] - p)
        return bool(abs(rel[0]) < PINCH_LATERAL and abs(rel[1]) < PINCH_LATERAL and abs(rel[2]) < TUBE_H / 2 - 0.005)

    def _attach(self):
        m, d, eq = self.model, self.data, self.info.weld_eq
        hand = m.eq_obj1id[eq]; tube = m.eq_obj2id[eq]
        R1 = d.xmat[hand].reshape(3, 3); relpos = R1.T @ (d.xpos[tube] - d.xpos[hand])
        q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, d.xquat[hand])
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, q1inv, d.xquat[tube])
        m.eq_data[eq, :3] = 0.0; m.eq_data[eq, 3:6] = relpos; m.eq_data[eq, 6:10] = relquat; m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1; self.attached = True

    def _detach(self):
        self.data.eq_active[self.info.weld_eq] = 0; self.attached = False

    def _update_weld(self, grip_cmd: float):
        if not self.attached and grip_cmd < 0.5 and self.ctrl.gripper_opening(self.data) < 0.5 and self._tube_in_pinch():
            self._attach()
        elif self.attached and grip_cmd > 0.5:
            self._detach()

    # ---- state ----------------------------------------------------------------------------
    def obs(self):
        p, R = self.ctrl.tcp_pose(self.data)
        return dict(
            qpos=self.data.qpos[:ARM_DOF + 2].copy(),
            qvel=self.data.qvel[:ARM_DOF + 2].copy(),
            tcp_pos=p, tcp_R=R,
            gripper=self.ctrl.gripper_opening(self.data),
            instruction=self.instruction, instruction_id=self.instruction_id,
            t=self.t, fill=self.fill_level(),
        )

    def fill_level(self) -> float:
        return self.liquid.receiver if self.liquid else 0.0

    def tube_pose(self):
        b = self.info.tube_body
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def tube_tilt_deg(self) -> float:
        return math.degrees(tube_tilt(self.data.xmat[self.info.tube_body]))

    def tube_speed(self) -> float:
        dof = self.model.body_dofadr[self.info.tube_body]
        return float(np.linalg.norm(self.data.qvel[dof:dof + 3]))

    def success(self) -> bool: raise NotImplementedError
    def failure_reason(self) -> str: raise NotImplementedError


class GraspRack(Task):
    name = "grasp"

    def success(self) -> bool:
        p, _ = self.tube_pose(); sc = self.info.slot_center
        in_xy = np.linalg.norm(p[:2] - sc[:2]) <= 0.010
        bottom = p[2] - TUBE_H / 2
        in_z = sc[2] - 0.005 <= bottom <= sc[2] + 0.015
        return bool(in_xy and in_z and self.tube_tilt_deg() <= 15 and self.ctrl.gripper_opening(self.data) > 0.5 and self.tube_speed() < 0.05)

    def failure_reason(self) -> str:
        p, _ = self.tube_pose(); sc = self.info.slot_center
        if p[2] < 0.02 and self.tube_tilt_deg() > 45: return "tube_knocked_over"
        if np.linalg.norm(p[:2] - sc[:2]) > 0.010: return f"xy_off_{np.linalg.norm(p[:2]-sc[:2])*1000:.0f}mm"
        if not (sc[2] - 0.005 <= p[2] - TUBE_H / 2 <= sc[2] + 0.015): return "not_in_slot_z"
        if self.tube_tilt_deg() > 15: return "tilted"
        if self.ctrl.gripper_opening(self.data) <= 0.5: return "not_released"
        return "moving"


class Insert(Task):
    name = "insert"

    def _task_reset(self, rng):
        # tube already in the gripper: tube centre at the TCP, fingers closed on it
        mujoco.mj_forward(self.model, self.data)
        p, R = self.ctrl.tcp_pose(self.data)
        reset_tube_pose(self.model, self.data, self.info, p - np.array([0, 0, INSERT_HOLD]), (1, 0, 0, 0))
        self.data.qpos[ARM_DOF:ARM_DOF + 2] = 0.0145
        self.data.ctrl[ARM_DOF] = 0.0                       # close
        mujoco.mj_forward(self.model, self.data)
        self._attach()

    def success(self) -> bool:
        p, _ = self.tube_pose(); sc = self.info.slot_center
        dx, dy = np.abs(p[:2] - sc[:2])
        bottom = p[2] - TUBE_H / 2
        seated = bottom <= sc[2] + 0.10 * self.info.slot_depth + 0.002
        return bool(dx < HOLDER_SLOT / 2 and dy < HOLDER_SLOT / 2 and seated and self.tube_tilt_deg() <= 10 and self.ctrl.gripper_opening(self.data) > 0.5)

    def failure_reason(self) -> str:
        p, _ = self.tube_pose(); sc = self.info.slot_center
        if self.tube_tilt_deg() > 45: return "dropped"
        if np.linalg.norm(p[:2] - sc[:2]) > HOLDER_SLOT / 2: return f"xy_off_{np.linalg.norm(p[:2]-sc[:2])*1000:.0f}mm"
        depth = (sc[2] + self.info.slot_depth) - (p[2] - TUBE_H / 2)
        if depth < 0.9 * self.info.slot_depth: return f"not_seated_{depth*1000:.0f}mm"
        if self.tube_tilt_deg() > 10: return "tilted"
        if self.ctrl.gripper_opening(self.data) <= 0.5: return "not_released"
        return "unknown"


class PourToLine(Task):
    name = "pour"
    TOL = 0.10          # ±10 % of the target fill fraction

    def _task_reset(self, rng):
        self.liquid = LiquidState(source=1.0, receiver=0.0)
        self._sync_liquid()

    def _sync_liquid(self):
        set_liquid_height(self.model, self.info.source_liquid_geom, -TUBE_H / 2 + TUBE_WALL, self.liquid.source * (TUBE_H - 2 * TUBE_WALL))
        set_liquid_height(self.model, self.info.receiver_liquid_geom, self.info.beaker_center[2] - self.data.xpos[self.info.fixture_body][2],
                          self.liquid.receiver * BEAKER_H)

    def opening_over_beaker(self) -> bool:
        b = self.info.tube_body
        pp = pour_point_world(self.data.xpos[b], self.data.xmat[b], TUBE_H / 2, TUBE_R - TUBE_WALL)
        return over_receiver(pp, self.info.beaker_center, self.info.beaker_r, self.info.beaker_h)

    def _task_step(self):
        tilt = tube_tilt(self.data.xmat[self.info.tube_body])
        self.liquid.step(tilt, self.opening_over_beaker(), 1.0 / 10.0)
        self._sync_liquid()

    def success(self) -> bool:
        m = self.info.mark_level
        return bool(abs(self.liquid.receiver - m) <= self.TOL * m and not self.liquid.spilled)

    def failure_reason(self) -> str:
        m = self.info.mark_level
        if self.liquid.spilled: return "spilled"
        if self.liquid.receiver < m * (1 - self.TOL): return f"underfilled_{self.liquid.receiver:.2f}_of_{m:.2f}"
        if self.liquid.receiver > m * (1 + self.TOL): return f"overfilled_{self.liquid.receiver:.2f}_of_{m:.2f}"
        return "unknown"


TASK_CLASSES = {"grasp": GraspRack, "pour": PourToLine, "insert": Insert}


def make_task(name: str, material: str = "opaque", randomize: bool = True) -> Task:
    return TASK_CLASSES[name](material, randomize)
