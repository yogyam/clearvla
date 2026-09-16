"""F3: Blender render cost via the in-process `bpy` module (no subprocess per frame).

Builds a minimal lab scene (table, rack, open tube with liquid), assigns either the
GLASS or OPAQUE material set from the design doc, and times N 224x224 frames per engine
with a small per-frame pose jitter (what a closed-loop observation step costs).
Pass condition: <= 0.5 s/frame. Also writes sample PNGs used by F4 (glass vs opaque).
"""
import argparse, math, time, json, sys
from pathlib import Path
import bpy
from mathutils import Vector

OUT = Path("results/week0/render")

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def principled(name, base=(0.8,0.8,0.8,1), rough=0.5, transmission=0.0, ior=1.45, alpha=1.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["Alpha"].default_value = alpha
    return m

MATS = {
    "glass":  dict(tube=dict(base=(1,1,1,1), rough=0.05, transmission=1.0, ior=1.5),
                   liquid=dict(base=(1,1,1,1), rough=0.0, transmission=1.0, ior=1.33)),
    "opaque": dict(tube=dict(base=(0.9,0.9,0.9,1), rough=0.6, transmission=0.0, ior=1.45),
                   liquid=dict(base=(0.05,0.2,0.8,1), rough=0.3, transmission=0.0, ior=1.33)),
}

def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

def build_scene(material):
    clear_scene()
    sc = bpy.context.scene
    # table
    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0,0,0)); table = bpy.context.object
    table.data.materials.append(principled("table", base=(0.28,0.30,0.33,1), rough=0.7))
    # rack (block with a hole is overkill for a benchmark: use a block)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.12,0.0,0.03)); rack = bpy.context.object
    rack.scale = (0.12,0.06,0.03); rack.data.materials.append(principled("rack", base=(0.2,0.2,0.22,1), rough=0.5))
    # tube: open cylinder, solidified to a thin wall
    bpy.ops.mesh.primitive_cylinder_add(radius=0.014, depth=0.10, location=(-0.05,0.0,0.05), end_fill_type='NOTHING', vertices=48)
    tube = bpy.context.object; tube.name = "tube"
    sol = tube.modifiers.new("solid", 'SOLIDIFY'); sol.thickness = 0.0015; sol.offset = 1.0
    bpy.ops.object.shade_smooth()
    # bottom disk so the tube isn't open at the base
    bpy.ops.mesh.primitive_cylinder_add(radius=0.0145, depth=0.0015, location=(-0.05,0.0,0.00075), vertices=48)
    bottom = bpy.context.object; bottom.name = "tube_bottom"; bottom.parent = tube; bottom.matrix_parent_inverse = tube.matrix_world.inverted()
    # liquid column (half-full)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.0125, depth=0.045, location=(-0.05,0.0,0.0015+0.0225), vertices=48)
    liq = bpy.context.object; liq.name = "liquid"; liq.parent = tube; liq.matrix_parent_inverse = tube.matrix_world.inverted()
    bpy.ops.object.shade_smooth()
    m = MATS[material]
    tm = principled("tube_mat", **m["tube"]); lm = principled("liquid_mat", **m["liquid"])
    for o in (tube, bottom): o.data.materials.append(tm)
    liq.data.materials.append(lm)
    # lights + world
    bpy.ops.object.light_add(type='AREA', location=(0.3,-0.4,0.8)); key = bpy.context.object
    key.data.energy = 25; key.data.size = 0.4; look_at(key, (0,0,0.05))
    bpy.ops.object.light_add(type='AREA', location=(-0.5,0.3,0.6)); fill = bpy.context.object
    fill.data.energy = 8; fill.data.size = 0.8; look_at(fill, (0,0,0.05))
    world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.5,0.55,0.6,1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.15
    # camera: front-oblique, like a fixed workspace cam
    cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam); cam.location = (0.0,-0.45,0.35); look_at(cam, (0.0,0.0,0.05))
    cam_data.lens = 35; sc.camera = cam
    # render settings
    sc.render.resolution_x = sc.render.resolution_y = 224; sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'
    sc.render.film_transparent = False
    return tube

def setup_engine(engine, samples):
    sc = bpy.context.scene
    engines = [e.identifier for e in sc.render.bl_rna.properties['engine'].enum_items]
    if engine == "cycles":
        sc.render.engine = 'CYCLES'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'; prefs.get_devices()
        for d in prefs.devices: d.use = (d.type == 'METAL')
        sc.cycles.device = 'GPU'; sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = True; sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 8; sc.cycles.transmission_bounces = 8; sc.cycles.transparent_max_bounces = 8
        sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
        return "CYCLES/METAL"
    eev = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
    sc.render.engine = eev; sc.eevee.taa_render_samples = samples
    try: sc.eevee.use_raytracing = True
    except Exception: pass
    return eev

def render_frames(tag, tube, n, jitter=True):
    sc = bpy.context.scene; times = []
    base = tube.location.copy()
    for i in range(n):
        if jitter:
            tube.location = base + Vector((0.002*math.sin(i*0.7), 0.002*math.cos(i*0.5), 0)); tube.rotation_euler.z = 0.05*i
        sc.render.filepath = str(OUT / f"{tag}_{i:03d}.png")
        t0 = time.perf_counter(); bpy.ops.render.render(write_still=True); times.append(time.perf_counter() - t0)
    return times

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--engines", default="cycles,eevee")
    ap.add_argument("--cycles-samples", type=int, default=32)
    ap.add_argument("--eevee-samples", type=int, default=16)
    a = ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for engine in a.engines.split(","):
        for material in ("glass", "opaque"):
            tube = build_scene(material)
            name = setup_engine(engine, a.cycles_samples if engine=="cycles" else a.eevee_samples)
            render_frames(f"{engine}_{material}_warm", tube, 1, jitter=False)   # first-frame compile cost, excluded
            ts = render_frames(f"{engine}_{material}", tube, a.frames)
            med = sorted(ts)[len(ts)//2]; mean = sum(ts)/len(ts)
            results[f"{engine}/{material}"] = dict(engine=name, frames=a.frames, mean_s=round(mean,3), median_s=round(med,3), max_s=round(max(ts),3))
            print(f"{engine:6s} {material:6s} [{name}]  mean {mean:.3f} s/frame  median {med:.3f}  max {max(ts):.3f}  -> {'PASS' if mean<=0.5 else 'FAIL'} (<=0.5)", flush=True)
    (OUT / "bench_render.json").write_text(json.dumps(dict(bpy=bpy.app.version_string, results=results), indent=2))
    print(json.dumps(results, indent=2))
if __name__ == "__main__": main()
