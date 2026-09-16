"""Scripted waypoint experts. They read true simulator state (oracle) and emit the same 10-D action a policy would.

Each expert is a small state machine: a list of phases, each with a target TCP pose, gripper command,
a max speed, and a completion predicate. Per-episode jitter (from the seed) varies approach heights so
the 150 demos per task are not clones.
"""
from __future__ import annotations
import math
import numpy as np
from envs.control import make_action
from envs.scene import TUBE_H, BEAKER_H
from envs.tasks import INSERT_HOLD
from envs.liquid import TILT_THRESH, RATE_K, VOL_RATIO

SPEED = 0.15            # m/s max TCP speed  (1.5 cm per 10 Hz step)
ROT_SPEED = math.radians(60)   # rad/s max rotation


def _rx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_axis(axis, theta):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * k @ k


def _slerp_R(R0, R1, max_angle):
    """Rotate from R0 toward R1 by at most max_angle (axis-angle)."""
    dR = R1 @ R0.T
    ang = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    if ang <= max_angle or ang < 1e-6: return R1
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * math.sin(ang))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = max_angle
    return (np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k) @ R0


class Expert:
    def __init__(self, task, rng: np.random.Generator):
        self.task, self.rng = task, rng
        self.phase = 0; self.hold = 0
        p, R = task.ctrl.tcp_pose(task.data)
        self.R_down = R.copy()                   # home orientation: gripper pointing straight down
        self.cmd_pos, self.cmd_R = p.copy(), R.copy()
        self.jit_h = rng.uniform(0.06, 0.10)     # approach height jitter
        self.jit_yaw = rng.uniform(-0.3, 0.3)    # grasp yaw jitter
        self.grasp_dz = rng.uniform(0.020, 0.030) # grasp above the tube centre so fingers clear fixture walls
        self.phases = self.build_phases()

    def build_phases(self): raise NotImplementedError

    def move_toward(self, target_pos, target_R, speed=SPEED, rot_speed=ROT_SPEED, dt=0.1):
        d = np.asarray(target_pos) - self.cmd_pos
        n = np.linalg.norm(d)
        self.cmd_pos = np.asarray(target_pos) if n <= speed * dt else self.cmd_pos + d / n * speed * dt
        self.cmd_R = _slerp_R(self.cmd_R, target_R, rot_speed * dt)

    def __call__(self, obs) -> np.ndarray:
        if self.phase >= len(self.phases):
            return make_action(self.cmd_pos, self.cmd_R, self.grip)
        ph = self.phases[self.phase]
        target_pos, target_R, self.grip = ph["target"](), ph.get("R", self.R_down), ph["grip"]
        self.move_toward(target_pos, target_R, ph.get("speed", SPEED), ph.get("rot_speed", ROT_SPEED))
        if ph["done"](obs, target_pos):
            self.hold += 1
            if self.hold >= ph.get("hold", 1): self.phase += 1; self.hold = 0
        return make_action(self.cmd_pos, self.cmd_R, self.grip)

    # common predicates
    def reached(self, obs, target, tol=0.006):
        return np.linalg.norm(obs["tcp_pos"] - target) < tol and np.linalg.norm(self.cmd_pos - target) < 1e-6

    def gripper_closed_on_tube(self, obs, target):
        return obs["gripper"] < 0.45 and self.reached(obs, target, 0.01)

    def gripper_open(self, obs, target):
        return obs["gripper"] > 0.85


class GraspRackExpert(Expert):
    def build_phases(self):
        t = self.task; tube0 = t.tube_pose()[0].copy(); tube = lambda: tube0
        R_g = _rz(self.jit_yaw) @ self.R_down
        above = lambda: tube() + [0, 0, self.jit_h]
        grasp = lambda: tube() + [0, 0, self.grasp_dz]
        self.lift_z = tube()[2] + 0.12
        sc = t.info.slot_center
        place_high = lambda: np.array([sc[0], sc[1], sc[2] + TUBE_H / 2 + self.grasp_dz + 0.10])
        place_low = lambda: np.array([sc[0], sc[1], sc[2] + TUBE_H / 2 + self.grasp_dz + 0.004])
        self._held = None
        def held_pos(): return np.array([self._held[0], self._held[1], self.lift_z])
        return [
            dict(target=above, R=R_g, grip=1.0, done=self.reached),
            dict(target=grasp, R=R_g, grip=1.0, done=self.reached, speed=0.08),
            dict(target=grasp, R=R_g, grip=0.0, done=self.gripper_closed_on_tube, hold=3),
            dict(target=lambda: np.array([self.cmd_pos[0], self.cmd_pos[1], self.lift_z]), R=R_g, grip=0.0, done=self.reached, speed=0.10),
            dict(target=place_high, R=R_g, grip=0.0, done=self.reached),
            dict(target=place_low, R=R_g, grip=0.0, done=self.reached, speed=0.06, hold=2),
            dict(target=place_low, R=R_g, grip=1.0, done=self.gripper_open, hold=3),
            dict(target=place_high, R=R_g, grip=1.0, done=self.reached),
        ]
    def _set_held(self):
        if self._held is None: self._held = self.task.tube_pose()[0].copy()


class InsertExpert(Expert):
    def build_phases(self):
        t = self.task; sc = t.info.slot_center; depth = t.info.slot_depth
        R_g = self.R_down
        high = lambda: np.array([sc[0], sc[1], sc[2] + depth + TUBE_H / 2 + INSERT_HOLD + 0.03])
        seat = lambda: np.array([sc[0], sc[1], sc[2] + TUBE_H / 2 + INSERT_HOLD + 0.002])
        def seated(obs, target):
            p, _ = t.tube_pose(); return (p[2] - TUBE_H / 2) <= sc[2] + 0.004 or self.reached(obs, target, 0.003)
        return [
            dict(target=lambda: np.array([self.cmd_pos[0], self.cmd_pos[1], max(self.cmd_pos[2], 0.25)]), R=R_g, grip=0.0, done=self.reached),
            dict(target=high, R=R_g, grip=0.0, done=self.reached),
            dict(target=seat, R=R_g, grip=0.0, done=seated, speed=0.04, hold=2),
            dict(target=lambda: self.cmd_pos.copy(), R=R_g, grip=1.0, done=self.gripper_open, hold=3),
            dict(target=high, R=R_g, grip=1.0, done=self.reached),
        ]


class PourExpert(Expert):
    TILT = math.radians(105)
    POUR_ROT = math.radians(35)      # slower rotation while holding the tube
    TILT_AXIS = 0                    # hand axis index to rotate about: 0 = x (pads lock the tube), 1 = y (torsional friction only)
    BASE_YAW = math.radians(-45)     # grasp yaw added to the per-episode jitter (x-axis tilt reachable, joint 7 mid-range)
    PRE_MARGIN = math.radians(25)    # pre-tilt this far below the pouring threshold

    def build_phases(self):
        t = self.task; tube0 = t.tube_pose()[0].copy(); tube = lambda: tube0
        R_g = _rz(self.BASE_YAW + self.jit_yaw) @ self.R_down
        Q = _rot_axis(R_g[:, self.TILT_AXIS], self.TILT)
        R_tilt = Q @ R_g
        above = lambda: tube() + [0, 0, self.jit_h]
        grasp = lambda: tube() + [0, 0, self.grasp_dz]
        bc = t.info.beaker_center; rim = bc[2] + BEAKER_H
        lift = lambda: np.array([self.cmd_pos[0], self.cmd_pos[1], rim + 0.15])   # commanded xy: no feedback via the held tube
        # Opening offset from the TCP, measured once after the lift (oracle) so it reflects how the tube actually
        # sits in the grip; then the TCP target for the tilted pose is static (no feedback through the held tube).
        from envs.liquid import opening_world
        target_opening = np.array([bc[0], bc[1], rim + 0.035])
        self._rel_top = None
        def rel_top():
            if self._rel_top is None:
                p, Rm = t.tube_pose(); tcp, Rh = t.ctrl.tcp_pose(t.data)
                self._rel_top = Rh.T @ (opening_world(p, Rm, TUBE_H / 2) - tcp)      # in the hand frame
            return self._rel_top
        pour_pos = lambda: target_opening - R_tilt @ rel_top()
        place = lambda: np.array([bc[0] - 0.10, bc[1] - 0.10, TUBE_H / 2 + self.grasp_dz + 0.003])
        place_high = lambda: place() + [0, 0, 0.10]
        self.mark = t.info.mark_level
        # stop early: untilting from TILT back past the threshold keeps pouring for (TILT - thresh)/POUR_ROT seconds
        dtheta = self.TILT - TILT_THRESH
        overshoot = 0.5 * RATE_K * dtheta * (dtheta / self.POUR_ROT)
        self.stop_at = self.mark - overshoot
        def poured(obs, target): return t.liquid.receiver >= self.stop_at
        def ang_to(Rt, obs): return math.acos(float(np.clip((np.trace(Rt.T @ obs["tcp_R"]) - 1) / 2, -1, 1)))
        tilted = lambda obs, tg: ang_to(R_tilt, obs) < math.radians(3)
        upright = lambda obs, tg: ang_to(R_g, obs) < math.radians(3)
        # pre-tilt just below the pouring threshold at a safe height, descend, then finish the tilt in place
        PRE = TILT_THRESH - self.PRE_MARGIN
        Qp = _rot_axis(R_g[:, self.TILT_AXIS], PRE); R_pre = Qp @ R_g
        pre_pour = lambda: pour_pos() + np.array([0, 0, 0.12])
        near = lambda obs, tg: ang_to(R_pre, obs) < math.radians(3)
        return [
            dict(target=above, R=R_g, grip=1.0, done=self.reached),
            dict(target=grasp, R=R_g, grip=1.0, done=self.reached, speed=0.08),
            dict(target=grasp, R=R_g, grip=0.0, done=self.gripper_closed_on_tube, hold=3),
            dict(target=lift, R=R_g, grip=0.0, done=self.reached, speed=0.10),
            dict(target=pre_pour, R=R_g, grip=0.0, done=self.reached),
            dict(target=pre_pour, R=R_pre, grip=0.0, done=near, rot_speed=self.POUR_ROT),
            dict(target=pour_pos, R=R_pre, grip=0.0, done=self.reached, speed=0.06),
            dict(target=pour_pos, R=R_tilt, grip=0.0, done=lambda o, tg: tilted(o, tg) or poured(o, tg), rot_speed=self.POUR_ROT),
            dict(target=pour_pos, R=R_tilt, grip=0.0, done=poured, rot_speed=self.POUR_ROT),
            dict(target=pour_pos, R=R_pre, grip=0.0, done=near, rot_speed=self.POUR_ROT),
            dict(target=pre_pour, R=R_pre, grip=0.0, done=self.reached),
            dict(target=pre_pour, R=R_g, grip=0.0, done=upright, rot_speed=self.POUR_ROT),
            dict(target=place_high, R=R_g, grip=0.0, done=self.reached),
            dict(target=place, R=R_g, grip=0.0, done=self.reached, speed=0.06, hold=2),
            dict(target=place, R=R_g, grip=1.0, done=self.gripper_open, hold=3),
            dict(target=place_high, R=R_g, grip=1.0, done=self.reached),
        ]


EXPERTS = {"grasp": GraspRackExpert, "insert": InsertExpert, "pour": PourExpert}


def make_expert(task, seed: int):
    return EXPERTS[task.name](task, np.random.default_rng(seed + 20_000))
