"""Scripted liquid: no fluid simulation.

The receiver fills at a rate proportional to how far the source tube is tilted past a threshold,
but only while the source opening is above the receiver mouth. Pouring anywhere else is a spill.
State is in fill fractions [0, 1] of each vessel. The paper states this approximation explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

# tube volume ≈ 47 mL, beaker (r=25 mm, h=50 mm) ≈ 98 mL -> a full tube fills the beaker to ~0.48 of the
# liquid column height... we use fractions of *height* and a volume ratio so conservation holds.
SOURCE_ML = math.pi * (0.0125 ** 2) * 0.097 * 1e6
RECEIVER_ML = math.pi * (0.0235 ** 2) * 0.050 * 1e6
VOL_RATIO = SOURCE_ML / RECEIVER_ML          # receiver height fraction gained per source height fraction lost

TILT_THRESH = math.radians(55.0)             # pouring starts past this tilt
RATE_K = 0.22                                # receiver fill fraction per (rad past threshold · second)


@dataclass
class LiquidState:
    source: float = 1.0        # fill fraction of source tube
    receiver: float = 0.0      # fill fraction of receiver beaker
    spilled: bool = False
    spilled_amount: float = 0.0
    trace: list = field(default_factory=list)

    def step(self, tilt_rad: float, opening_over_receiver: bool, dt: float):
        rate = RATE_K * max(0.0, tilt_rad - TILT_THRESH)
        if rate <= 0 or self.source <= 0:
            self.trace.append(self.receiver); return
        d_recv = rate * dt
        d_src = min(self.source, d_recv / VOL_RATIO)
        d_recv = d_src * VOL_RATIO
        self.source -= d_src
        if opening_over_receiver and self.receiver + d_recv <= 1.0:
            self.receiver += d_recv
        else:
            self.spilled = True; self.spilled_amount += d_recv
            self.receiver = min(1.0, self.receiver + (d_recv if opening_over_receiver else 0.0))
        self.trace.append(self.receiver)


def tube_tilt(xmat: np.ndarray) -> float:
    """Angle between the tube's local +z axis and world up, radians."""
    z = np.asarray(xmat).reshape(3, 3)[:, 2]
    return math.acos(float(np.clip(z[2], -1.0, 1.0)))


def opening_world(xpos, xmat, half_height: float) -> np.ndarray:
    """World position of the tube's open end (local +z top, centre of the mouth)."""
    return np.asarray(xpos) + np.asarray(xmat).reshape(3, 3)[:, 2] * half_height


def pour_point_world(xpos, xmat, half_height: float, radius: float) -> np.ndarray:
    """Where liquid actually leaves a tilted tube: the lowest point of the mouth's rim. That is the mouth centre
    displaced by `radius` along the horizontal projection of the tube axis (the side the tube leans toward)."""
    R = np.asarray(xmat).reshape(3, 3); axis = R[:, 2]; centre = np.asarray(xpos) + axis * half_height
    h = np.array([axis[0], axis[1], 0.0]); n = np.linalg.norm(h)
    if n < 1e-6: return centre                      # upright: no preferred side
    return centre + (h / n) * radius - np.array([0, 0, radius * n])   # rim point also sits lower than the centre


def over_receiver(pour_pt: np.ndarray, beaker_center: np.ndarray, beaker_r: float, beaker_h: float) -> bool:
    """The pour point must fall inside the beaker's inner wall and above 90 % of its height."""
    dxy = np.linalg.norm(pour_pt[:2] - beaker_center[:2])
    return bool(dxy < beaker_r and pour_pt[2] > beaker_center[2] + beaker_h * 0.9)
