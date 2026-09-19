"""MuJoCo -> Blender (in-process bpy) bridge, generic over the scene builder.

BlenderBridge(model, info, material) builds lights, world and camera once; rebind(model, info, material)
rebuilds all geometry from the compiled MjModel tables (robot meshes from mesh_vert, primitives from
geom_type/size). Special cases for realism: the tube is a hollow open cylinder, the beaker is a hollow
cylinder with fill-mark rings, liquids are cylinders resized each frame from the live geom sizes.
The camera is copied from the MuJoCo camera so Blender RGB and MuJoCo masks are pixel aligned.
"""
from __future__ import annotations
import math
import numpy as np
import mujoco
import bpy
from mathutils import Vector, Quaternion, Matrix

MATS = {
    "glass":  dict(tube=dict(base=(1, 1, 1, 1), rough=0.05, transmission=1.0, ior=1.5),
                   liquid=dict(base=(1, 1, 1, 1), rough=0.0, transmission=1.0, ior=1.33)),
    "opaque": dict(tube=dict(base=(0.9, 0.9, 0.9, 1), rough=0.6, transmission=0.0, ior=1.45),
                   liquid=dict(base=(0.05, 0.2, 0.8, 1), rough=0.3, transmission=0.0, ior=1.33)),
}
MATS["alpha"] = MATS["glass"]
TUBE_WALL = 0.0015
BEAKER_WALL = 0.002


def _principled(name, base=(0.8, 0.8, 0.8, 1), rough=0.5, transmission=0.0, ior=1.45, emission=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = base; b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior; b.inputs["Transmission Weight"].default_value = transmission
    if emission is not None:
        b.inputs["Emission Color"].default_value = emission; b.inputs["Emission Strength"].default_value = 0.4
    if transmission > 0.5:
        # Glass with caustics disabled casts an opaque black shadow, and you see that shadow *through* the object.
        # Route shadow rays to a Transparent BSDF so glass shadows are light, as in the real world.
        out = nt.nodes["Material Output"]; lp = nt.nodes.new("ShaderNodeLightPath"); tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr.inputs[0].default_value = (0.9, 0.9, 0.9, 1)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs[0]); nt.links.new(b.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def _look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def _link(obj):
    bpy.context.scene.collection.objects.link(obj); return obj


def _mesh_obj(name, verts, faces, smooth=True):
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    if smooth:
        for p in me.polygons: p.use_smooth = True
    return _link(bpy.data.objects.new(name, me))


def _cylinder_verts(r, z0, z1, n=48):
    ring = lambda z: [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    return ring(z0) + ring(z1)


def _closed_cylinder(name, r, h, n=48):
    """Solid cylinder centred at origin (like a MuJoCo cylinder geom, h = full height)."""
    v = _cylinder_verts(r, -h / 2, h / 2, n) + [(0, 0, -h / 2), (0, 0, h / 2)]
    f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    f += [(2 * n, (i + 1) % n, i) for i in range(n)] + [(2 * n + 1, n + i, n + (i + 1) % n) for i in range(n)]
    return _mesh_obj(name, v, f)


def _hollow_cylinder(name, r_out, r_in, z0, z1, n=48, bottom=True):
    """Open-top vessel: outer wall, inner wall, rim, and a floor of thickness (z_floor)."""
    vo = _cylinder_verts(r_out, z0, z1, n); vi = _cylinder_verts(r_in, z0 + (TUBE_WALL if bottom else 0), z1, n)
    v = vo + vi; O, I = 0, 2 * n
    f = [(O + i, O + (i + 1) % n, O + n + (i + 1) % n, O + n + i) for i in range(n)]                     # outer
    f += [(I + n + i, I + n + (i + 1) % n, I + (i + 1) % n, I + i) for i in range(n)]                    # inner (flipped)
    f += [(O + n + i, O + n + (i + 1) % n, I + n + (i + 1) % n, I + n + i) for i in range(n)]            # rim
    if bottom:
        v += [(0, 0, z0)]; c = len(v) - 1
        f += [(c, O + (i + 1) % n, O + i) for i in range(n)]                                             # outer floor
        v += [(0, 0, z0 + TUBE_WALL)]; c2 = len(v) - 1
        f += [(c2, I + i, I + (i + 1) % n) for i in range(n)]                                            # inner floor
    return _mesh_obj(name, v, f)


def _torus(name, R, r, n=48, m=8):
    v = [((R + r * math.cos(2 * math.pi * j / m)) * math.cos(2 * math.pi * i / n),
          (R + r * math.cos(2 * math.pi * j / m)) * math.sin(2 * math.pi * i / n),
          r * math.sin(2 * math.pi * j / m)) for i in range(n) for j in range(m)]
    f = [(i * m + j, ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m) for i in range(n) for j in range(m)]
    return _mesh_obj(name, v, f)


class BlenderBridge:
    def __init__(self, model, info, material="opaque", res=224, engine="cycles", samples=32):
        self.res = res
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        sc.render.resolution_x = sc.render.resolution_y = res; sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'
        self._lights_world()
        cam_data = bpy.data.cameras.new("cam"); self.cam = _link(bpy.data.objects.new("cam", cam_data)); sc.camera = self.cam
        cam_data.sensor_fit = 'VERTICAL'
        self._engine(engine, samples)
        self.geom_objs = []; self.body_objs = {}; self.liquids = {}
        self.rebind(model, info, material)

    # ---- static -------------------------------------------------------------------------------
    def _lights_world(self):
        bpy.ops.object.light_add(type='AREA', location=(0.8, -0.6, 1.2)); self.key = bpy.context.object
        self.key.data.energy = 50; self.key.data.size = 0.6; _look_at(self.key, (0.5, 0, 0.05))
        bpy.ops.object.light_add(type='AREA', location=(-0.2, 0.6, 1.0)); self.fill = bpy.context.object
        self.fill.data.energy = 15; self.fill.data.size = 1.0; _look_at(self.fill, (0.5, 0, 0.05))
        w = bpy.data.worlds.new("w"); bpy.context.scene.world = w; w.use_nodes = True
        w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.55, 0.6, 1)
        w.node_tree.nodes["Background"].inputs[1].default_value = 0.15

    def _engine(self, engine, samples):
        sc = bpy.context.scene
        if engine == "cycles":
            sc.render.engine = 'CYCLES'
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'METAL'; prefs.get_devices()
            for d in prefs.devices: d.use = (d.type == 'METAL')
            sc.cycles.device = 'GPU'; sc.cycles.samples = samples; sc.cycles.use_adaptive_sampling = True
            sc.cycles.use_denoising = True; sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 8
            sc.cycles.transparent_max_bounces = 8; sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
        else:
            ids = [e.identifier for e in sc.render.bl_rna.properties['engine'].enum_items]
            sc.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in ids else 'BLENDER_EEVEE'
            sc.eevee.taa_render_samples = samples

    # ---- per-episode geometry ------------------------------------------------------------------
    def rebind(self, model, info, material="opaque"):
        self.model, self.info, self.material = model, info, material
        for o in self.geom_objs + list(self.body_objs.values()):
            bpy.data.objects.remove(o, do_unlink=True)
        self.geom_objs, self.body_objs, self.liquids = [], {}, {}
        # purge the datablocks the removed objects leave behind (meshes, materials, images); without this every
        # rebind leaks ~75 meshes + materials and a 450-episode render grows into gigabytes of swap
        for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
            for blk in [b for b in coll if b.users == 0]:
                try: coll.remove(blk)
                except Exception: pass
        mats = MATS[material]
        self.m_tube = _principled("tube_mat", **mats["tube"]); self.m_liquid = _principled("liquid_mat", **mats["liquid"])
        self.m_table = _principled("table", base=(0.28, 0.30, 0.33, 1), rough=0.7)
        self.m_fixture = _principled("fixture", base=(0.2, 0.2, 0.22, 1), rough=0.5)
        self.m_mark = _principled("mark", base=(0.85, 0.15, 0.15, 1), rough=0.4, emission=(0.85, 0.15, 0.15, 1))
        self.mj_mats = {}
        self.cams = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("front", "wrist")}
        self.cams = {n: i for n, i in self.cams.items() if i >= 0}
        li = getattr(info, "light_intensity", 1.0)
        self.key.data.energy = 50 * li; self.fill.data.energy = 15 * li
        mesh_cache = {}
        done_beaker = False
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            gtype = model.geom_type[g]; b = int(model.geom_bodyid[g]); size = model.geom_size[g]
            # skip collision-only robot meshes and beaker box pieces (replaced by a proper hollow cylinder)
            if gtype == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] != 2: continue
            if name.startswith("beaker_wall") or name == "beaker_floor" or name.startswith("beaker_mark") or name.startswith("beaker_omark"):
                if not done_beaker:
                    self._beaker(model, info, b); done_beaker = True
                continue
            parent = self._body(b)
            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, 0)); o = bpy.context.object; o.data.materials.append(self.m_table)
                self.geom_objs.append(o); continue
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                if mid not in mesh_cache:
                    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]; fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                    me = bpy.data.meshes.new(f"mjmesh_{mid}"); me.from_pydata(model.mesh_vert[va:va + vn].tolist(), [], model.mesh_face[fa:fa + fn].tolist()); me.update()
                    for p in me.polygons: p.use_smooth = True
                    mesh_cache[mid] = me
                o = _link(bpy.data.objects.new(name, mesh_cache[mid]))
                matid = model.geom_matid[g]
                if matid >= 0:
                    if matid not in self.mj_mats: self.mj_mats[matid] = _principled(f"mjmat_{matid}", base=tuple(model.mat_rgba[matid]), rough=0.5)
                    if not o.data.materials: o.data.materials.append(self.mj_mats[matid])
            elif name == "tube_geom":
                r, h = size[0], 2 * size[1]
                o = _hollow_cylinder(name, r, r - TUBE_WALL, -h / 2, h / 2); o.data.materials.append(self.m_tube)
            elif name.endswith("_liquid"):
                o = _closed_cylinder(name, size[0], 1.0); o.data.materials.append(self.m_liquid)   # unit height, scaled per frame
                self.liquids[g] = o
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                o = _closed_cylinder(name, size[0], 2 * size[1]); o.data.materials.append(self.m_fixture)
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0)); o = bpy.context.object; o.name = name
                o.scale = (size[0], size[1], size[2]); o.data.materials.append(self.m_mark if "mark" in name else self.m_fixture)
            else:
                continue
            o.parent = parent; o.location = Vector(model.geom_pos[g])
            o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Quaternion(model.geom_quat[g])
            self.geom_objs.append(o)
        self.n_objects = len(self.geom_objs)

    def _body(self, b):
        if b not in self.body_objs:
            e = _link(bpy.data.objects.new(f"body_{b}", None)); e.rotation_mode = 'QUATERNION'; self.body_objs[b] = e
        return self.body_objs[b]

    def _beaker(self, model, info, b):
        from envs.scene import BEAKER_R, BEAKER_H, BEAKER_WALL as BW
        parent = self._body(b)
        floor_t = 0.0015
        o = _hollow_cylinder("beaker", BEAKER_R + BW, BEAKER_R, 0.0, 2 * floor_t + BEAKER_H); o.data.materials.append(self.m_tube)
        o.parent = parent; self.geom_objs.append(o)
        zm = 2 * floor_t + info.mark_level * BEAKER_H
        for nm, R in (("mark_in", BEAKER_R - 0.0004), ("mark_out", BEAKER_R + BW + 0.0004)):
            t = _torus(nm, R, 0.0008); t.data.materials.append(self.m_mark); t.parent = parent; t.location = (0, 0, zm); self.geom_objs.append(t)

    # ---- per-frame ----------------------------------------------------------------------------
    def sync(self, data, camera="front", **_):
        m = self.model; cid = self.cams[camera]
        self.cam.data.angle_y = math.radians(m.cam_fovy[cid])
        for b, e in self.body_objs.items():
            e.location = Vector(data.xpos[b]); e.rotation_quaternion = Quaternion(data.xquat[b])
        for g, o in self.liquids.items():
            h = 2 * m.geom_size[g, 1]
            o.scale = (1, 1, max(h, 1e-4)); o.location = Vector(m.geom_pos[g])
        R = Matrix(np.array(data.cam_xmat[cid]).reshape(3, 3).tolist())
        self.cam.matrix_world = Matrix.Translation(Vector(data.cam_xpos[cid])) @ R.to_4x4()

    def render(self, path: str):
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)


class MaskRenderer:
    """MuJoCo segmentation mask for the task-critical geoms, pixel-aligned with the Blender camera."""
    def __init__(self, model, info, res=224, camera="front"):
        self.r = mujoco.Renderer(model, height=res, width=res); self.r.enable_segmentation_rendering()
        self.cam = camera; self.crit = np.array(info.critical_geoms); self.tube = info.tube_geom
    def __call__(self, data):
        self.r.update_scene(data, camera=self.cam); seg = self.r.render()
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        return (np.isin(seg[..., 0], self.crit) & is_geom), ((seg[..., 0] == self.tube) & is_geom)
