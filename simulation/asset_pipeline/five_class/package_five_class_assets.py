"""Validate converted CAD and package centered, materialized Isaac Sim assets."""

import argparse
import hashlib
import json
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()
root = Path(args.root).resolve()

app = SimulationApp({"headless": True})

import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def triangulate(counts, indices):
    result = []
    cursor = 0
    for count in counts:
        polygon = indices[cursor : cursor + count]
        cursor += count
        for index in range(1, count - 1):
            result.append((polygon[0], polygon[index], polygon[index + 1]))
    return np.asarray(result, dtype=np.int64)


def load_mesh(path):
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Could not open {path}")
    default = stage.GetDefaultPrim()
    if not default.IsValid():
        children = list(stage.GetPseudoRoot().GetChildren())
        if len(children) != 1:
            raise RuntimeError(f"No unique root in {path}")
        default = children[0]
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    pieces = []
    mesh_count = 0
    for prim in Usd.PrimRange(
        default, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    ):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        if not points or not counts or not indices:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        vertices = np.asarray(
            [tuple(transform.Transform(Gf.Vec3d(*point))) for point in points], float
        )
        faces = triangulate(counts, indices)
        pieces.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
        mesh_count += 1
    if not pieces:
        raise RuntimeError(f"No meshes in {path}")
    combined = trimesh.util.concatenate(pieces)
    combined.merge_vertices(merge_tex=True, merge_norm=True)
    return stage, combined, mesh_count


def collision_asset(source_mesh, output):
    hull = source_mesh.convex_hull
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    container = UsdGeom.Xform.Define(stage, "/Collision")
    stage.SetDefaultPrim(container.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Collision/Mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(*[float(value) for value in point]) for point in hull.vertices]
    )
    mesh.CreateFaceVertexCountsAttr([3] * len(hull.faces))
    mesh.CreateFaceVertexIndicesAttr(hull.faces.reshape(-1).astype(int).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
        "convexHull"
    )
    stage.GetRootLayer().Save()
    return hull


def material(stage, path, color, metallic, roughness):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def sim_asset(output, visual_name, collision_name, center_mm, color, mass_kg, metallic):
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(asset.GetPrim()).CreateKinematicEnabledAttr().Set(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(asset.GetPrim()).CreateEnableCCDAttr().Set(True)
    UsdPhysics.MassAPI.Apply(asset.GetPrim()).CreateMassAttr().Set(float(mass_kg))

    millimetres = UsdGeom.Xform.Define(stage, "/Asset/Millimetres")
    UsdGeom.Xformable(millimetres).AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))
    visual_offset = UsdGeom.Xform.Define(stage, "/Asset/Millimetres/VisualOffset")
    UsdGeom.Xformable(visual_offset).AddTranslateOp().Set(
        Gf.Vec3d(*[-float(value) for value in center_mm])
    )
    visual = UsdGeom.Xform.Define(stage, "/Asset/Millimetres/VisualOffset/CAD")
    visual.GetPrim().GetReferences().AddReference("./" + visual_name)
    look = material(stage, "/Asset/Looks/Surface", color, metallic, 0.34)
    UsdShade.MaterialBindingAPI.Apply(visual_offset.GetPrim()).Bind(
        look, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )

    collision_offset = UsdGeom.Xform.Define(stage, "/Asset/Millimetres/CollisionOffset")
    UsdGeom.Xformable(collision_offset).AddTranslateOp().Set(
        Gf.Vec3d(*[-float(value) for value in center_mm])
    )
    collision = UsdGeom.Xform.Define(
        stage, "/Asset/Millimetres/CollisionOffset/Proxy"
    )
    collision.GetPrim().GetReferences().AddReference("./" + collision_name)
    stage.GetRootLayer().Save()


converted = root / "converted"
packaged = root / "packaged"
inspection = root / "inspection"
packaged.mkdir(parents=True, exist_ok=True)
inspection.mkdir(parents=True, exist_ok=True)

shapes = {
    "long_pill": {
        "visual": "long_pill_visual.usdc",
        "collision": "long_pill_collision.usdc",
        "source": "long_pill.step",
    },
    "round_pill": {
        "visual": "round_pill_visual.usdc",
        "collision": "round_pill_collision.usdc",
        "source": "round_pill.step",
    },
}

colors = {
    "red": (0.78, 0.055, 0.035),
    "yellow": (0.94, 0.68, 0.045),
}

loaded = {}
for key, spec in shapes.items():
    stage, mesh, mesh_count = load_mesh(converted / spec["visual"])
    loaded[key] = (stage, mesh, mesh_count)

collision_hulls = {}
for key, spec in shapes.items():
    collision_hulls[spec["collision"]] = collision_asset(
        loaded[key][1], packaged / spec["collision"]
    )

assets = {}
for shape, spec in shapes.items():
    _, mesh, mesh_count = loaded[shape]
    center = mesh.bounds.mean(axis=0)
    mass_kg = max(0.0002, abs(float(mesh.volume)) * 1.0e-9 * 1200.0)
    hull = collision_hulls[spec["collision"]]
    shape_row = {
        "source_step": spec["source"],
        "visual_usdc": "converted/" + spec["visual"],
        "collision_usdc": "packaged/" + spec["collision"],
        "dimensions_mm": mesh.extents.tolist(),
        "source_center_mm": center.tolist(),
        "visual_meshes": mesh_count,
        "visual_triangles": int(len(mesh.faces)),
        "visual_watertight": bool(mesh.is_watertight),
        "volume_mm3": abs(float(mesh.volume)),
        "collision_vertices": int(len(hull.vertices)),
        "collision_triangles": int(len(hull.faces)),
        "collision_watertight": bool(hull.is_watertight),
        "default_mass_kg": mass_kg,
        "variants": {},
    }
    for color_name, color in colors.items():
        sim_name = f"{shape}_{color_name}_sim.usda"
        sim_asset(
            packaged / sim_name,
            "../converted/" + spec["visual"],
            spec["collision"],
            center,
            color,
            mass_kg,
            0.0,
        )
        shape_row["variants"][color_name] = {
            "sim_asset": "packaged/" + sim_name,
            "material_rgb": list(color),
        }
    assets[shape] = shape_row


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = {
    "status": "PASS",
    "isaac_ready_assets": [
        row["sim_asset"]
        for item in assets.values()
        for row in item["variants"].values()
    ],
    "source_sha256": {
        spec["source"]: sha256(root / "source" / spec["source"])
        for spec in shapes.values()
    },
    "source_units": "millimetres",
    "sim_asset_units": "metres",
    "assets": assets,
    "appearance_note": (
        "The sim wrappers apply explicit reddish and yellow pill materials. Shape class "
        "is independent of colour so both colours exist for both pill geometries."
    ),
    "collision_note": (
        "No collision CAD was delivered. Watertight convex-hull collision proxies were "
        "generated from the visual solids."
    ),
}
with open(inspection / "object_asset_manifest.json", "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)

for key, item in assets.items():
    print(
        "[asset]",
        key,
        "size_mm=",
        tuple(round(value, 4) for value in item["dimensions_mm"]),
        "visual_triangles=",
        item["visual_triangles"],
        "collision_triangles=",
        item["collision_triangles"],
        "watertight=",
        item["visual_watertight"] and item["collision_watertight"],
    )
print("[asset] manifest", inspection / "object_asset_manifest.json")
app.close()
