"""Programmatic scene builder (MjSpec). One function builds every task/material/seed.

build_scene(task, material, seed) -> (MjModel, SceneInfo)

Design rules:
- Everything the tube can enter (rack, holder, beaker) is built from convex box pieces, because
  MuJoCo mesh collision uses the convex hull.
- Liquids are non-colliding cylinders whose height is set at runtime (see envs/liquid.py).
- The material only changes rgba here; true glass comes from render/bridge.py (Blender).
- All randomisation is drawn from `seed`, so seed k gives identical geometry for every material.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = ROOT / "envs" / "franka" / "panda.xml"

TASKS = ("grasp", "pour", "insert")
MATERIALS = ("opaque", "glass", "alpha")

# geometry (metres)
TUBE_R, TUBE_H = 0.014, 0.10           # test tube: radius, full height
TUBE_WALL = 0.0015
RACK_SLOT, RACK_DEPTH, WALL_T = 0.040, 0.040, 0.006     # grasp&rack: loose slot
HOLDER_SLOT, HOLDER_DEPTH = 0.030, 0.060                # insert: 1 mm clearance per side
BEAKER_R, BEAKER_H, BEAKER_WALL, BEAKER_SEGS = 0.025, 0.050, 0.002, 12
LIQUID_MARGIN = 0.0015                                  # liquid radius = inner radius - margin
TCP_OFFSET = 0.1034                                     # hand frame -> fingertip centre (m)
GRIP_SCALE = 8.0                                        # gripper actuator gain/bias multiplier (Menagerie default peaks at ~4 N)
NOSLIP_ITERS = 0                                        # MuJoCo noslip solver iterations: removes friction creep of held objects

# MuJoCo rgba per material. glass/alpha look the same in MuJoCo; Blender makes the difference.
RGBA = {
    "opaque": dict(tube=(0.90, 0.90, 0.90, 1.0), liquid=(0.05, 0.20, 0.80, 1.0)),
    "glass":  dict(tube=(0.90, 0.90, 0.90, 0.30), liquid=(0.55, 0.70, 0.90, 0.30)),
    "alpha":  dict(tube=(0.90, 0.90, 0.90, 0.30), liquid=(0.55, 0.70, 0.90, 0.30)),
}
RGBA_TABLE = (0.28, 0.30, 0.33, 1.0)
RGBA_RACK = (0.20, 0.20, 0.22, 1.0)
RGBA_MARK = (0.85, 0.15, 0.15, 1.0)

# nominal layout; randomisation is added on top
NOMINAL = dict(
    tube=np.array([0.50, -0.08, TUBE_H / 2 + 0.001]),
    source=np.array([0.50, -0.08, TUBE_H / 2 + 0.001]),
    rack=np.array([0.55, 0.14, 0.0]),
    holder=np.array([0.55, 0.10, 0.0]),
    beaker=np.array([0.55, 0.10, 0.0]),
    cam_pos=np.array([0.86, -0.52, 0.50]),   # diagonal agent view: beaker stays visible beside the hand during the pour
    cam_lookat=np.array([0.53, 0.02, 0.06]),
    cam_fovy=36.0,
    light_intensity=1.0,
)
RANDOM = dict(object_xy=0.05, fixture_xy=0.03, light=0.20, cam_pos=0.01, cam_deg=2.0)


@dataclass
class SceneInfo:
    task: str
    material: str
    seed: int
    tube_body: int = -1
    tube_geom: int = -1
    tube_qposadr: int = -1
    tcp_site: int = -1
    cam_id: int = -1
    fixture_body: int = -1            # rack / holder / beaker
    slot_center: np.ndarray | None = None   # world xyz of slot floor centre (rack/holder)
    slot_depth: float = 0.0
    beaker_center: np.ndarray | None = None
    beaker_r: float = BEAKER_R
    beaker_h: float = BEAKER_H
    mark_level: float = 0.5           # target fill fraction (pour)
    source_liquid_geom: int = -1
    receiver_liquid_geom: int = -1
    critical_geoms: list[int] = field(default_factory=list)   # GT "task-critical object" for H1/H5 masks
    params: dict = field(default_factory=dict)                 # every sampled random value
    cam_pos: np.ndarray | None = None
    cam_xyaxes: np.ndarray | None = None
    light_intensity: float = 1.0
    weld_eq: int = -1


def _lookat_xyaxes(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(-f, x)             # camera looks down -Z; Z = -f
    return np.concatenate([x, y])


def _rot_z(deg):
    a = math.radians(deg); return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _add_box(body, name, pos, half, rgba, quat=None, collide=True, group=0):
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(half), rgba=list(rgba), group=group)
    if quat is not None: g.quat = list(quat)
    if not collide: g.contype = 0; g.conaffinity = 0
    return g


def _add_slot_fixture(body, prefix, slot, depth, rgba):
    """Square slot open at the top: floor + 4 walls. Returns floor top z (local)."""
    h = slot / 2; t = WALL_T / 2; floor_t = 0.003
    _add_box(body, f"{prefix}_floor", (0, 0, floor_t), (h + WALL_T, h + WALL_T, floor_t), rgba)
    z = 2 * floor_t + depth / 2
    _add_box(body, f"{prefix}_wall_px", ( h + t, 0, z), (t, h + WALL_T, depth / 2), rgba)
    _add_box(body, f"{prefix}_wall_nx", (-h - t, 0, z), (t, h + WALL_T, depth / 2), rgba)
    _add_box(body, f"{prefix}_wall_py", (0,  h + t, z), (h, t, depth / 2), rgba)
    _add_box(body, f"{prefix}_wall_ny", (0, -h - t, z), (h, t, depth / 2), rgba)
    return 2 * floor_t


def _add_beaker(body, prefix, r, height, wall, segs, rgba_glass, mark_level, rgba_liquid):
    """Open cylinder from `segs` box segments + floor, a fill-mark ring on the inner wall, and a liquid cylinder."""
    floor_t = 0.0015
    body.add_geom(name=f"{prefix}_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, floor_t],
                  size=[r + wall, floor_t, 0], rgba=list(rgba_glass))
    chord = math.pi * (r + wall / 2) / segs
    for i in range(segs):
        a = 2 * math.pi * i / segs
        pos = ((r + wall / 2) * math.cos(a), (r + wall / 2) * math.sin(a), 2 * floor_t + height / 2)
        quat = (math.cos(a / 2), 0, 0, math.sin(a / 2))
        _add_box(body, f"{prefix}_wall{i}", pos, (wall / 2, chord * 1.02, height / 2), rgba_glass, quat=quat)
    # fill mark: thin ring of dark segments just inside the wall (visual only)
    zm = 2 * floor_t + mark_level * height
    for i in range(segs):
        a = 2 * math.pi * i / segs
        quat = (math.cos(a / 2), 0, 0, math.sin(a / 2))
        pos_in = ((r - 0.0004) * math.cos(a), (r - 0.0004) * math.sin(a), zm)
        _add_box(body, f"{prefix}_mark{i}", pos_in, (0.0004, chord * 1.02, 0.0015), RGBA_MARK, quat=quat, collide=False, group=2)
        pos_out = ((r + wall + 0.0004) * math.cos(a), (r + wall + 0.0004) * math.sin(a), zm)
        _add_box(body, f"{prefix}_omark{i}", pos_out, (0.0004, chord * 1.02, 0.0010), RGBA_MARK, quat=quat, collide=False, group=2)
    liq = body.add_geom(name=f"{prefix}_liquid", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 2 * floor_t + 0.0005],
                        size=[r - LIQUID_MARGIN, 0.0005, 0], rgba=list(rgba_liquid))
    liq.contype = 0; liq.conaffinity = 0
    return liq, 2 * floor_t


def _add_tube(spec_body, name, pos, rgba_tube, rgba_liquid, with_liquid, fill=0.0):
    b = spec_body.add_body(name=name, pos=list(pos))
    b.add_freejoint(name=f"{name}_free")
    g = b.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[TUBE_R, TUBE_H / 2, 0],
                   rgba=list(rgba_tube), mass=0.02, friction=[1.5, 0.05, 0.001], condim=4,
                   solref=[0.005, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    liq = None
    if with_liquid:
        h = max(fill * (TUBE_H - 2 * TUBE_WALL), 0.001)
        liq = b.add_geom(name=f"{name}_liquid", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         pos=[0, 0, -TUBE_H / 2 + TUBE_WALL + h / 2], size=[TUBE_R - TUBE_WALL - 0.0005, h / 2, 0],
                         rgba=list(rgba_liquid))
        liq.contype = 0; liq.conaffinity = 0
    return b, g, liq


def build_scene(task: str, material: str, seed: int, randomize: bool = True):
    assert task in TASKS and material in MATERIALS
    rng = np.random.default_rng(seed)
    P = {}
    def jit(key, nominal, amp):
        v = np.asarray(nominal, float).copy()
        if randomize:
            d = rng.uniform(-amp, amp, size=2); v[:2] += d; P[key] = d.tolist()
        return v

    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.modelname = f"clearvla_{task}_{material}_{seed}"
    for b in spec.bodies:
        if b.name.startswith('link') or b.name in ('hand', 'left_finger', 'right_finger'): b.gravcomp = 1.0
    # Menagerie's gripper actuator peaks at ~4 N; a real Panda grips with 20-70 N. Scale gain and bias 8x
    # (ctrl range stays 0-255) so a tilted tube cannot lever the fingers apart.
    grip = [a for a in spec.actuators if a.name == "actuator8"][0]
    grip.gainprm[0] *= GRIP_SCALE; grip.biasprm[1] *= GRIP_SCALE; grip.biasprm[2] *= GRIP_SCALE
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.noslip_iterations = NOSLIP_ITERS
    spec.visual.global_.offwidth = 640; spec.visual.global_.offheight = 640
    wb = spec.worldbody
    rgba = RGBA[material]

    # table + lights
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=list(RGBA_TABLE))
    li = NOMINAL["light_intensity"] * (1 + (rng.uniform(-RANDOM["light"], RANDOM["light"]) if randomize else 0.0))
    P["light_intensity"] = li
    l = wb.add_light(name="key", pos=[0.8, -0.6, 1.2], dir=[-0.3, 0.5, -0.8])
    l.diffuse = [0.7 * li] * 3; l.specular = [0.1] * 3; l.castshadow = True
    f = wb.add_light(name="fill", pos=[-0.2, 0.6, 1.0], dir=[0.5, -0.4, -0.7])
    f.diffuse = [0.3 * li] * 3; f.specular = [0.0] * 3; f.castshadow = False

    # camera with small pose jitter
    cam_pos = np.asarray(NOMINAL["cam_pos"], float).copy(); lookat = np.asarray(NOMINAL["cam_lookat"], float).copy()
    if randomize:
        dp = rng.uniform(-RANDOM["cam_pos"], RANDOM["cam_pos"], size=3); cam_pos += dp
        dyaw = rng.uniform(-RANDOM["cam_deg"], RANDOM["cam_deg"]); lookat = cam_pos + _rot_z(dyaw) @ (lookat - cam_pos)
        P["cam_dpos"] = dp.tolist(); P["cam_dyaw"] = dyaw
    xyaxes = _lookat_xyaxes(cam_pos, lookat)
    cam = wb.add_camera(name="front", pos=list(cam_pos), fovy=NOMINAL["cam_fovy"])
    # xyaxes -> quaternion
    x, y = xyaxes[:3], xyaxes[3:]; z = np.cross(x, y); R = np.stack([x, y, z], 1)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); cam.quat = q.tolist()

    # TCP site on the hand
    hand = spec.body("hand"); hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0, 0], rgba=[0, 1, 0, 0.0])

    info = SceneInfo(task=task, material=material, seed=seed)
    if task == "grasp":
        tube_pos = jit("tube", NOMINAL["tube"], RANDOM["object_xy"])
        _add_tube(wb, "tube", tube_pos, rgba["tube"], rgba["liquid"], with_liquid=False)
        rp = jit("rack", NOMINAL["rack"], RANDOM["fixture_xy"])
        rack = wb.add_body(name="rack", pos=list(rp))
        floor_top = _add_slot_fixture(rack, "rack", RACK_SLOT, RACK_DEPTH, RGBA_RACK)
        info.slot_center = rp + np.array([0, 0, floor_top]); info.slot_depth = RACK_DEPTH
    elif task == "insert":
        # tube starts in the gripper (placed by the task at reset); initial pos is a placeholder above the table
        _add_tube(wb, "tube", [0.45, -0.15, 0.30], rgba["tube"], rgba["liquid"], with_liquid=False)
        hp = jit("holder", NOMINAL["holder"], RANDOM["fixture_xy"])
        holder = wb.add_body(name="holder", pos=list(hp))
        floor_top = _add_slot_fixture(holder, "holder", HOLDER_SLOT, HOLDER_DEPTH, RGBA_RACK)
        info.slot_center = hp + np.array([0, 0, floor_top]); info.slot_depth = HOLDER_DEPTH
    elif task == "pour":
        sp = jit("source", NOMINAL["source"], RANDOM["object_xy"])
        _, _, sliq = _add_tube(wb, "tube", sp, rgba["tube"], rgba["liquid"], with_liquid=True, fill=1.0)
        bp = jit("beaker", NOMINAL["beaker"], RANDOM["fixture_xy"])
        mark = float(rng.uniform(0.45, 0.60)) if randomize else 0.5; P["mark_level"] = mark
        beaker = wb.add_body(name="beaker", pos=list(bp))
        rliq, floor_top = _add_beaker(beaker, "beaker", BEAKER_R, BEAKER_H, BEAKER_WALL, BEAKER_SEGS, rgba["tube"], mark, rgba["liquid"])
        info.beaker_center = bp + np.array([0, 0, floor_top]); info.mark_level = mark

    # sticky gripper: weld hand<->tube, inactive until the task attaches it on a closed grasp (see envs/tasks.py)
    eq = spec.add_equality(name="grasp_weld", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1="hand", name2="tube", active=False)
    eq.solref = [0.004, 1.0]
    # extend the Franka 'home' keyframe with the tube's free joint so mj_resetDataKeyframe places it correctly
    tube_spawn = [float(v) for v in spec.body("tube").pos]
    spec.keys[0].qpos = list(spec.keys[0].qpos) + tube_spawn + [1, 0, 0, 0]
    model = spec.compile()
    info.tube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tube")
    info.tube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tube_geom")
    info.tube_qposadr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tube_free")])
    info.tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    info.weld_eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
    info.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    fixture = {"grasp": "rack", "insert": "holder", "pour": "beaker"}[task]
    info.fixture_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fixture)
    if task == "pour":
        info.source_liquid_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tube_liquid")
        info.receiver_liquid_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "beaker_liquid")
        info.critical_geoms = [info.receiver_liquid_geom] + [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"beaker_{k}{i}") for i in range(BEAKER_SEGS) for k in ("mark", "omark")]
    elif task == "grasp":
        info.critical_geoms = [info.tube_geom]
    else:
        info.critical_geoms = [info.tube_geom] + [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"holder_wall_{s}") for s in ("px", "nx", "py", "ny")]
    info.params = P; info.cam_pos = cam_pos; info.cam_xyaxes = xyaxes; info.light_intensity = li
    return model, info


def set_liquid_height(model: mujoco.MjModel, geom_id: int, base_z_local: float, height: float):
    """Resize a liquid cylinder geom in place (local frame of its body). Call mj_forward afterwards."""
    h = max(height, 1e-4)
    model.geom_size[geom_id, 1] = h / 2
    model.geom_pos[geom_id, 2] = base_z_local + h / 2


def reset_tube_pose(model, data, info: SceneInfo, pos, quat=(1, 0, 0, 0)):
    a = info.tube_qposadr
    data.qpos[a:a + 3] = pos; data.qpos[a + 3:a + 7] = quat
    data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tube_free")]:][:6] = 0


def home(model, data):
    mujoco.mj_resetDataKeyframe(model, data, 0)


if __name__ == "__main__":
    for t in TASKS:
        m, i = build_scene(t, "opaque", 0)
        print(t, "nbody", m.nbody, "ngeom", m.ngeom, "critical", len(i.critical_geoms), "params", i.params)
