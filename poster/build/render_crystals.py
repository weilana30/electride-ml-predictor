"""Rebuild the crystals inside electride_crystals_v2.blend from real structures.

Keeps the file's collections, root empties, camera, lights and world, and replaces
only the contents of each Host framework and Electron regions collection.

    Blender -b electride_crystals_v2.blend --python render_crystals.py -- GEOMETRY_DIR OUT_DIR SAMPLES [save]
"""
import json, math, sys
from pathlib import Path
import bpy, numpy as np
from mathutils import Euler, Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
GEOM, OUT, SAMPLES = Path(argv[0]), Path(argv[1]), int(argv[2])
SAVE = len(argv) > 3 and argv[3] == "save"
ONLY = argv[4].split(",") if len(argv) > 4 else None
OUT.mkdir(parents=True, exist_ok=True)

# Poster palette, sRGB
COLORS = {"cation": "#9d9795", "second": "#555960", "anion": "#1c1c1e", "void": "#daaa00", "frame": "#6f727b"}
ROLE = {"Ca": "cation", "Y": "cation", "Ba": "cation", "Al": "second", "Na": "second",
        "O": "anion", "N": "anion", "Si": "anion"}
# Drawn atom radius in angstrom: small enough to see the void between atoms
DRAW = {"Ca": .56, "Y": .5, "Ba": .78, "Al": .36, "Na": .52, "O": .36, "N": .48, "Si": .38}
SPIN = {"0D": (-8, 0, 30), "2D": (-19, 0, 8), "3D": (-8, 0, 30)}  # z spin first, then x tilt
FIT_W, FIT_H = 3.2, 3.0
FIT_EXTRA = {"0D": .96}  # spheres near the cell frame
# 1D lies on its side: channels run left to right and recede, so they read as parallel rods
TURN = {"1D": Matrix.Rotation(math.radians(-28), 3, "Z") @ Matrix.Rotation(math.radians(90), 3, "Y")}
CAGE_RADIUS = 1.6  # angstrom, 0D spheres

def linear(h):
    c = [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return [(x / 12.92 if x <= .04045 else ((x + .055) / 1.055) ** 2.4) for x in c] + [1]

def material(name, color, rough, source=None, coat=0.0):
    m = bpy.data.materials.get(source) if source else None
    m = m or bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.name = name; m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = linear(color)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = 0
    if "Coat Weight" in b.inputs: b.inputs["Coat Weight"].default_value = coat
    return m

mats = {"void": material("Void regions - orange", COLORS["void"], .38, "Anionic electron regions - gold", .25),
        "cation": material("Atoms - cation blue", COLORS["cation"], .42, "Host atoms - gray", .2),
        "second": material("Atoms - secondary blue", COLORS["second"], .42, coat=.2),
        "anion": material("Atoms - anion navy", COLORS["anion"], .42, coat=.2),
        "frame": material("Cell and bonds", COLORS["frame"], .55, "Framework - soft gray")}

cam = bpy.data.objects[next(o.name for o in bpy.data.objects if o.type == "CAMERA")]
print("CAMERA", tuple(round(v, 3) for v in cam.location), tuple(round(math.degrees(v), 1) for v in cam.rotation_euler))
bpy.context.view_layer.update()
right = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))
up = cam.matrix_world.to_3x3() @ Vector((0, 1, 0))

sphere_cache = {}
def sphere(el, s):
    key = el
    if key in sphere_cache: return sphere_cache[key]
    import bmesh
    me = bpy.data.meshes.new(f"{el} sphere"); bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=24, radius=DRAW[el] * s)
    bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    me.materials.append(mats[ROLE[el]]); sphere_cache[key] = me
    return me

def rod(name, a, b, r, coll, root):
    a, b = Vector(a), Vector(b); d = b - a
    me = bpy.data.meshes.new(name)
    n = 12; vs = []; fs = []
    z = d.normalized(); x = z.orthogonal().normalized(); y = z.cross(x)
    for i in range(n):
        t = 2 * math.pi * i / n; o = (x * math.cos(t) + y * math.sin(t)) * r
        vs += [a + o, b + o]
        j = (i + 1) % n; fs.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    me.from_pydata([tuple(v) for v in vs], [], fs)
    for p in me.polygons: p.use_smooth = True
    me.materials.append(mats["frame"])
    o = bpy.data.objects.new(name, me); coll.objects.link(o); o.parent = root
    return o

for label in ("0D", "1D", "2D", "3D"):
    model = bpy.data.collections[f"{label} - model"]
    root = next(o for o in bpy.data.objects if o.name.startswith(f"{label} - move"))
    subs = list(model.children)
    host = next(c for c in subs if c.name.startswith("Host"))
    regions = next(c for c in subs if c.name.startswith("Electron"))
    for c in (host, regions):
        for o in list(c.objects): bpy.data.objects.remove(o, do_unlink=True)

    info = json.loads((GEOM / f"{label}.json").read_text())
    g = np.load(GEOM / f"{label}.npz")
    corners = np.array(info["corners"]); center = corners.mean(0)
    if label in TURN:
        root.rotation_mode = "QUATERNION"; root.rotation_quaternion = TURN[label].to_quaternion()
        R = TURN[label]
    else:
        root.rotation_mode = "ZYX"; root.rotation_euler = Euler([math.radians(v) for v in SPIN[label]], "ZYX")
        R = root.rotation_euler.to_matrix()
    pts = [R @ Vector(p - center) for p in corners]
    w = max(p.dot(right) for p in pts) - min(p.dot(right) for p in pts)
    h = max(p.dot(up) for p in pts) - min(p.dot(up) for p in pts)
    s = min(FIT_W / w, FIT_H / h) * FIT_EXTRA.get(label, 1)
    print(label, info["name"], "scale", round(s, 3))

    for i, (el, x, y, z) in enumerate(info["atoms"]):
        o = bpy.data.objects.new(f"{el} {i:03d}", sphere(el, s))
        o.location = (np.array((x, y, z)) - center) * s
        host.objects.link(o); o.parent = root
    pos = [(np.array(a[1:]) - center) * s for a in info["atoms"]]
    for i, (a, b) in enumerate(info["bonds"]):
        rod(f"Bond {i:03d}", pos[a], pos[b], .012, host, root)
    c = (corners - center) * s
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j > i: rod(f"Cell edge {i}-{j}", c[i], c[j], .016, host, root)

    if info.get("void_spheres"):
        import bmesh
        me = bpy.data.meshes.new("Cage sphere"); bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=40, radius=CAGE_RADIUS * s)
        bm.to_mesh(me); bm.free()
        for p in me.polygons: p.use_smooth = True
        me.materials.append(mats["void"])
        for i, c0 in enumerate(info["void_spheres"]):
            o = bpy.data.objects.new(f"Cage {i + 1:02d}", me)
            o.location = (np.array(c0) - center) * s
            regions.objects.link(o); o.parent = root
    else:
        me = bpy.data.meshes.new(f"{info['name']} void surface")
        verts = (g["verts"] - center) * s
        me.from_pydata(verts.tolist(), [], g["faces"].tolist())
        me.update()
        for p in me.polygons: p.use_smooth = True
        me.materials.append(mats["void"])
        o = bpy.data.objects.new(f"{info['name']} void regions", me); regions.objects.link(o); o.parent = root
    sphere_cache.clear()

sc = bpy.context.scene
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"

def show(only):
    for key in ("0D", "1D", "2D", "3D"):
        bpy.data.collections[f"{key} - model"].hide_render = only is not None and key != only
    bpy.context.view_layer.update()

loc0, scale0 = cam.location.copy(), cam.data.ortho_scale
sc.render.resolution_x, sc.render.resolution_y = 3200, 690
show(None)
if not ONLY or "strip" in ONLY:
    sc.render.filepath = str(OUT / "electride_crystals_v2_transparent.png"); bpy.ops.render.render(write_still=True)
sc.render.resolution_x = sc.render.resolution_y = 1600
for label in ("0D", "1D", "2D", "3D"):
    if ONLY and label not in ONLY: continue
    show(label)
    root = next(o for o in bpy.data.objects if o.name.startswith(f"{label} - move"))
    cam.location = loc0 + right * (root.location.dot(right) - loc0.dot(right))
    cam.data.ortho_scale = 3.5
    sc.render.filepath = str(OUT / f"electride_{label.lower()}_v2_transparent.png")
    bpy.ops.render.render(write_still=True)
cam.location, cam.data.ortho_scale = loc0, scale0
show(None)
sc.render.resolution_x, sc.render.resolution_y = 3200, 690
if SAVE: bpy.ops.wm.save_as_mainfile(filepath=str(Path(argv[1]) / "electride_crystals_v2.blend"))
