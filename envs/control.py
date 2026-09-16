"""Control interface: 10-D action (TCP xyz + 6D rotation + gripper) at 10 Hz -> joint position targets.

Damped-least-squares IK on the `tcp` site with a nullspace bias toward the home configuration.
Both the scripted expert and learned policies go through `apply_action`, so recorded actions are exactly
what a policy must emit.
"""
from __future__ import annotations
import numpy as np
import mujoco

ARM_DOF = 7
CTRL_HZ = 10.0
DAMPING = 0.05
IK_ITERS = 6
NULLSPACE_GAIN = 0.05
MAX_DQ = 0.35            # rad per control step (joint speed cap)


def rot6d_from_mat(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]])


def mat_from_rot6d(r6: np.ndarray) -> np.ndarray:
    a, b = r6[:3], r6[3:6]
    x = a / (np.linalg.norm(a) + 1e-9)
    y = b - np.dot(x, b) * x; y /= (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def make_action(pos, R, gripper_open: float) -> np.ndarray:
    return np.concatenate([np.asarray(pos, float), rot6d_from_mat(np.asarray(R, float)), [float(gripper_open)]])


def split_action(a: np.ndarray):
    return a[:3], mat_from_rot6d(a[3:9]), float(np.clip(a[9], 0, 1))


class Controller:
    def __init__(self, model: mujoco.MjModel, tcp_site: int):
        self.model, self.site = model, tcp_site
        self.scratch = mujoco.MjData(model)
        self.q_home = model.key_qpos[0][:ARM_DOF].copy()
        self.lo, self.hi = model.jnt_range[:ARM_DOF, 0], model.jnt_range[:ARM_DOF, 1]
        self.substeps = int(round(1.0 / CTRL_HZ / model.opt.timestep))
        self._jacp = np.zeros((3, model.nv)); self._jacr = np.zeros((3, model.nv))

    def tcp_pose(self, data):
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, data, target_pos, target_R, q_init=None) -> np.ndarray:
        q = (data.qpos[:ARM_DOF] if q_init is None else q_init).copy()
        s = self.scratch; s.qpos[:] = data.qpos
        tq = np.zeros(4); mujoco.mju_mat2Quat(tq, np.asarray(target_R, float).reshape(-1))
        for _ in range(IK_ITERS):
            s.qpos[:ARM_DOF] = q; mujoco.mj_kinematics(self.model, s); mujoco.mj_comPos(self.model, s)
            p = s.site_xpos[self.site]; Rm = s.site_xmat[self.site]
            sq = np.zeros(4); mujoco.mju_mat2Quat(sq, Rm)
            err_rot = np.zeros(3); mujoco.mju_subQuat(err_rot, tq, sq)   # local frame of the site
            err_rot = Rm.reshape(3, 3) @ err_rot                          # -> world frame, to match jacr
            err = np.concatenate([np.asarray(target_pos) - p, err_rot])
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3: break
            mujoco.mj_jacSite(self.model, s, self._jacp, self._jacr, self.site)
            J = np.vstack([self._jacp[:, :ARM_DOF], self._jacr[:, :ARM_DOF]])
            JJt = J @ J.T + (DAMPING ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # nullspace bias toward home
            Jp = J.T @ np.linalg.solve(JJt, J)
            dq += (np.eye(ARM_DOF) - Jp) @ (NULLSPACE_GAIN * (self.q_home - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def apply_action(self, data, action: np.ndarray):
        """Set ctrl from a 10-D action. Caller steps physics `self.substeps` times."""
        pos, R, g = split_action(np.asarray(action, float))
        q_t = self.ik(data, pos, R)
        dq = np.clip(q_t - data.qpos[:ARM_DOF], -MAX_DQ, MAX_DQ)
        data.ctrl[:ARM_DOF] = data.qpos[:ARM_DOF] + dq
        data.ctrl[ARM_DOF] = 255.0 * g

    def step(self, data, action: np.ndarray):
        self.apply_action(data, action)
        for _ in range(self.substeps): mujoco.mj_step(self.model, data)

    def gripper_opening(self, data) -> float:
        """0 = closed, 1 = fully open (0.04 m per finger)."""
        return float(np.clip((data.qpos[ARM_DOF] + data.qpos[ARM_DOF + 1]) / 0.08, 0, 1))
