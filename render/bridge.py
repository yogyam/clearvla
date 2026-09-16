"""MuJoCo -> Blender (in-process bpy) bridge.

Builds the Blender scene once (table, rack, tube+liquid, robot visual meshes, lights, camera),
then per observation copies body poses from MjData and renders. Camera intrinsics/extrinsics
come from the MuJoCo camera so Blender RGB and MuJoCo segmentation masks are pixel-aligned.
"""
import math
from pathlib import Path
import numpy as np
import mujoco
import bpy
from mathutils import Vector, Quaternion, Matrix

MATS = {
    "glass":  dict(tube=dict(base=(1,1,1,1), rough=0.05, transmission=1.0, ior=1.5),
                   liquid=dict(base=(1,1,1,1), rough=0.0, transmission=1.0, ior=1.33)),
    "opaque": dict(tube=dict(base=(0.9,0.9,0.9,1), rough=0.6, transmission=0.0, ior=1.45),
                   liquid=dict(base=(0.05,0.2,0.8,1), rough=0.3, transmission=0.0, ior=1.33)),
}

def _principled(name, base=(0.8,0.8,0.8,1), rough=0.5, transmission=0.0, ior=1.45):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = base; b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior; b.inputs["Transmission Weight"].default_value = transmission
    return m

def _look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

class BlenderBridge:
    def __init__(self, model: mujoco.MjModel, material: str, asset_dir: Path, camera: str = "front",
                 res: int = 224, engine: str = "cycles", samples: int = 32, tube_body: str = "tube", fill: float = 0.5):
        self.model, self.res, self.cam_name = model, res, camera
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        self._static(material, fill)
        self.body_objs = self._import_robot(model, asset_dir)
        self.tube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tube_body)
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        cam_data = bpy.data.cameras.new("cam"); self.cam = bpy.data.objects.new("cam", cam_data)
        sc.collection.objects.link(self.cam); sc.camera = self.cam
        cam_data.sensor_fit = 'VERTICAL'; cam_data.angle_y = math.radians(model.cam_fovy[self.cam_id])
        sc.render.resolution_x = sc.render.resolution_y = res; sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'
        self._engine(engine, samples)

    def _static(self, material, fill):
        bpy.ops.mesh.primitive_plane_add(size=3, location=(0,0,0)); t = bpy.context.object
        t.data.materials.append(_principled("table", base=(0.28,0.30,0.33,1), rough=0.7))
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0.62,0.15,0.03)); r = bpy.context.object
        r.scale = (0.12,0.06,0.06); r.data.materials.append(_principled("rack", base=(0.2,0.2,0.22,1), rough=0.5))
        # tube root empty; children carry the geometry so a single transform moves everything
        self.tube = bpy.data.objects.new("tube_root", None); bpy.context.scene.collection.objects.link(self.tube)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.014, depth=0.10, location=(0,0,0), end_fill_type='NOTHING', vertices=48)
        wall = bpy.context.object; sol = wall.modifiers.new("solid", 'SOLIDIFY'); sol.thickness = 0.0015; sol.offset = 1.0
        bpy.ops.object.shade_smooth()
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0145, depth=0.0015, location=(0,0,-0.05+0.00075), vertices=48); bottom = bpy.context.object
        h = 0.097 * fill
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0125, depth=h, location=(0,0,-0.05+0.0015+h/2), vertices=48); liq = bpy.context.object
        bpy.ops.object.shade_smooth(); self.liquid = liq
        m = MATS[material]; tm = _principled("tube_mat", **m["tube"]); lm = _principled("liquid_mat", **m["liquid"])
        for o in (wall, bottom): o.data.materials.append(tm); o.parent = self.tube
        liq.data.materials.append(lm); liq.parent = self.tube
        bpy.ops.object.light_add(type='AREA', location=(0.8,-0.6,1.2)); k = bpy.context.object
        k.data.energy = 50; k.data.size = 0.6; _look_at(k, (0.5,0,0.05))
        bpy.ops.object.light_add(type='AREA', location=(-0.2,0.6,1.0)); f = bpy.context.object
        f.data.energy = 15; f.data.size = 1.0; _look_at(f, (0.5,0,0.05))
        w = bpy.data.worlds.new("w"); bpy.context.scene.world = w; w.use_nodes = True
        w.node_tree.nodes["Background"].inputs[0].default_value = (0.5,0.55,0.6,1)
        w.node_tree.nodes["Background"].inputs[1].default_value = 0.15

    def _import_robot(self, model, asset_dir: Path):
        """One empty per MuJoCo body; visual-group mesh geoms imported once and parented with their local offset."""
        body_objs = {}
        mat_cache = {}
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2: continue
            b = model.geom_bodyid[g]
            if b not in body_objs:
                e = bpy.data.objects.new(f"body_{b}", None); bpy.context.scene.collection.objects.link(e); body_objs[b] = e
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
            path = asset_dir / f"{mesh_name}.obj"
            if not path.exists(): continue
            before = set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=str(path), forward_axis='Y', up_axis='Z')
            new = [o for o in bpy.data.objects if o not in before]
            mid = model.geom_matid[g]
            if mid >= 0 and mid not in mat_cache:
                rgba = tuple(model.mat_rgba[mid]); mat_cache[mid] = _principled(f"mjmat_{mid}", base=rgba, rough=0.5)
            for o in new:
                o.parent = body_objs[b]; o.location = Vector(model.geom_pos[g])
                o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Quaternion(model.geom_quat[g])
                if mid >= 0:
                    o.data.materials.clear(); o.data.materials.append(mat_cache[mid])
                for p in o.data.polygons: p.use_smooth = True
        for e in body_objs.values(): e.rotation_mode = 'QUATERNION'
        self.n_robot_meshes = sum(1 for o in bpy.data.objects if o.type == "MESH" and o.parent in body_objs.values())
        return body_objs

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

    def sync(self, data: mujoco.MjData, fill: float | None = None):
        for b, e in self.body_objs.items():
            e.location = Vector(data.xpos[b]); e.rotation_quaternion = Quaternion(data.xquat[b])
        self.tube.rotation_mode = 'QUATERNION'
        self.tube.location = Vector(data.xpos[self.tube_id]); self.tube.rotation_quaternion = Quaternion(data.xquat[self.tube_id])
        if fill is not None:
            h = max(0.097 * fill, 1e-4); self.liquid.scale.z = h / self.liquid.dimensions.z * self.liquid.scale.z if self.liquid.dimensions.z else 1
        # camera: MuJoCo cam frame looks down -Z with +Y up, same as Blender
        R = Matrix(np.array(data.cam_xmat[self.cam_id]).reshape(3,3).tolist())
        self.cam.matrix_world = Matrix.Translation(Vector(data.cam_xpos[self.cam_id])) @ R.to_4x4()

    def render(self, path: str):
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)

class MaskRenderer:
    """MuJoCo segmentation mask for the tube, pixel-aligned with the Blender camera."""
    def __init__(self, model, res=224, camera="front", tube_geom="tube_geom"):
        self.r = mujoco.Renderer(model, height=res, width=res); self.r.enable_segmentation_rendering()
        self.cam = camera; self.gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tube_geom)
    def __call__(self, data):
        self.r.update_scene(data, camera=self.cam); seg = self.r.render()
        return (seg[..., 0] == self.gid) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
