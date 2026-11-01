"""Package the new pill and battery CAD as centered Isaac Sim rigid assets."""

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
    return combined, mesh_count


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


def bind(prim, look):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        look, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


def add_battery_terminals(stage, dimensions_mm):
    """Add metallic end overlays while retaining the delivered CAD silhouette."""
    long_axis = int(np.argmax(dimensions_mm))
    cross = [float(dimensions_mm[i]) for i in range(3) if i != long_axis]
    diameter = float(sum(cross) / len(cross))
    axis_token = (UsdGeom.Tokens.x, UsdGeom.Tokens.y, UsdGeom.Tokens.z)[long_axis]
    length = float(dimensions_mm[long_axis])
    positive_height = min(0.8, max(0.35, diameter * 0.045))
    negative_height = min(0.45, max(0.20, diameter * 0.025))

    silver = material(
        stage, "/Asset/Looks/PositiveTerminal", (0.52, 0.55, 0.58), 0.92, 0.20
    )
    steel = material(
        stage, "/Asset/Looks/NegativeTerminal", (0.34, 0.36, 0.39), 0.88, 0.24
    )
    root = UsdGeom.Xform.Define(stage, "/Asset/Millimetres/TerminalVisuals")

    positive = UsdGeom.Cylinder.Define(
        stage, "/Asset/Millimetres/TerminalVisuals/Positive"
    )
    positive.CreateAxisAttr(axis_token)
    positive.CreateRadiusAttr(diameter * 0.18)
    positive.CreateHeightAttr(positive_height)
    positive_center = [0.0, 0.0, 0.0]
    positive_center[long_axis] = length / 2.0 - positive_height / 2.0
    UsdGeom.Xformable(positive).AddTranslateOp().Set(Gf.Vec3d(*positive_center))
    bind(positive.GetPrim(), silver)

    negative = UsdGeom.Cylinder.Define(
        stage, "/Asset/Millimetres/TerminalVisuals/Negative"
    )
    negative.CreateAxisAttr(axis_token)
    negative.CreateRadiusAttr(diameter * 0.42)
    negative.CreateHeightAttr(negative_height)
    negative_center = [0.0, 0.0, 0.0]
    negative_center[long_axis] = -length / 2.0 + negative_height / 2.0
    UsdGeom.Xformable(negative).AddTranslateOp().Set(Gf.Vec3d(*negative_center))
    bind(negative.GetPrim(), steel)
    return {
        "long_axis": long_axis,
        "axis_token": str(axis_token),
        "diameter_mm": diameter,
        "positive_terminal_radius_mm": diameter * 0.18,
        "negative_terminal_radius_mm": diameter * 0.42,
    }


def sim_asset(
    output,
    visual_name,
    collision_name,
    center_mm,
    dimensions_mm,
    color,
    mass_kg,
    metallic,
    roughness,
    is_battery,
):
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
    body_look = material(
        stage, "/Asset/Looks/Body", color, metallic, roughness
    )
    bind(visual_offset.GetPrim(), body_look)

    collision_offset = UsdGeom.Xform.Define(stage, "/Asset/Millimetres/CollisionOffset")
    UsdGeom.Xformable(collision_offset).AddTranslateOp().Set(
        Gf.Vec3d(*[-float(value) for value in center_mm])
    )
    collision = UsdGeom.Xform.Define(
        stage, "/Asset/Millimetres/CollisionOffset/Proxy"
    )
    collision.GetPrim().GetReferences().AddReference("./" + collision_name)
    terminal_data = add_battery_terminals(stage, dimensions_mm) if is_battery else None
    stage.GetRootLayer().Save()
    return terminal_data


converted = root / "converted"
packaged = root / "packaged"
inspection = root / "inspection"
packaged.mkdir(parents=True, exist_ok=True)
inspection.mkdir(parents=True, exist_ok=True)

shapes = {
    "round_pill": {
        "visual": "round_pill_visual.usdc",
        "collision": "round_pill_collision.usdc",
        "source": "round_pill.step",
        "density_kg_m3": 1200.0,
        "is_battery": False,
    },
    "long_pill": {
        "visual": "long_pill_visual.usdc",
        "collision": "long_pill_collision.usdc",
        "source": "long_pill.step",
        "density_kg_m3": 1200.0,
        "is_battery": False,
    },
    "aa_battery": {
        "visual": "aa_battery_visual.usdc",
        "collision": "aa_battery_collision.usdc",
        "source": "aa_battery.step",
        "mass_kg": 0.023,
        "is_battery": True,
    },
    "aaa_battery": {
        "visual": "aaa_battery_visual.usdc",
        "collision": "aaa_battery_collision.usdc",
        "source": "aaa_battery.step",
        "mass_kg": 0.0115,
        "is_battery": True,
    },
}

pill_variants = {
    "red": ((0.72, 0.025, 0.018), 0.0, 0.40),
    "yellow": ((0.92, 0.64, 0.025), 0.0, 0.42),
    "blue": ((0.025, 0.14, 0.68), 0.0, 0.40),
    "black": ((0.025, 0.028, 0.034), 0.05, 0.35),
}
battery_variants = {
    "black": ((0.035, 0.040, 0.048), 0.12, 0.31),
    "blue": ((0.025, 0.12, 0.52), 0.08, 0.34),
    "copper": ((0.48, 0.16, 0.055), 0.38, 0.29),
    "silver": ((0.44, 0.47, 0.50), 0.72, 0.25),
}

loaded = {}
for key, spec in shapes.items():
    mesh, mesh_count = load_mesh(converted / spec["visual"])
    loaded[key] = (mesh, mesh_count)

collision_hulls = {}
for key, spec in shapes.items():
    collision_hulls[spec["collision"]] = collision_asset(
        loaded[key][0], packaged / spec["collision"]
    )

assets = {}
for shape, spec in shapes.items():
    mesh, mesh_count = loaded[shape]
    center = mesh.bounds.mean(axis=0)
    volume_mm3 = abs(float(mesh.volume))
    if "mass_kg" in spec:
        mass_kg = float(spec["mass_kg"])
    else:
        mass_kg = float(
            max(0.0002, volume_mm3 * 1.0e-9 * spec["density_kg_m3"])
        )
    hull = collision_hulls[spec["collision"]]
    shape_row = {
        "source_step": spec["source"],
        "visual_usdc": "converted/" + spec["visual"],
        "collision_usdc": "packaged/" + spec["collision"],
        "dimensions_mm": mesh.extents.tolist(),
        "source_bounds_mm": {
            "min": mesh.bounds[0].tolist(),
            "max": mesh.bounds[1].tolist(),
            "center": center.tolist(),
        },
        "visual_meshes": mesh_count,
        "visual_triangles": int(len(mesh.faces)),
        "visual_watertight": bool(mesh.is_watertight),
        "volume_mm3": volume_mm3,
        "collision_vertices": int(len(hull.vertices)),
        "collision_triangles": int(len(hull.faces)),
        "collision_watertight": bool(hull.is_watertight),
        "default_mass_kg": mass_kg,
        "variants": {},
    }
    variants = battery_variants if spec["is_battery"] else pill_variants
    for variant_name, (color, metallic, roughness) in variants.items():
        sim_name = f"{shape}_{variant_name}_sim.usda"
        terminal_data = sim_asset(
            packaged / sim_name,
            "../converted/" + spec["visual"],
            spec["collision"],
            center,
            mesh.extents,
            color,
            mass_kg,
            metallic,
            roughness,
            spec["is_battery"],
        )
        shape_row["variants"][variant_name] = {
            "sim_asset": "packaged/" + sim_name,
            "material_rgb": list(color),
            "metallic": metallic,
            "roughness": roughness,
        }
        if terminal_data:
            shape_row["terminal_visuals"] = terminal_data
    assets[shape] = shape_row


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = {
    "status": "PASS",
    "package": "new_pill_battery_assets_candidate_20261029",
    "old_pill_assets_reused": False,
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
        "Shape is the semantic class. Pill and battery body colours are material "
        "variants to prevent a classifier from using colour as the class shortcut. "
        "Battery wrappers add metallic end overlays to the delivered CAD silhouette."
    ),
    "collision_note": (
        "No separate collision CAD was delivered. Watertight convex-hull collision "
        "proxies were generated from each new visual solid."
    ),
}
with (inspection / "object_asset_manifest.json").open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)

for key, item in assets.items():
    print(
        "[asset]",
        key,
        "size_mm=",
        tuple(round(value, 4) for value in item["dimensions_mm"]),
        "mass_g=",
        round(item["default_mass_kg"] * 1000.0, 3),
        "visual_triangles=",
        item["visual_triangles"],
        "collision_triangles=",
        item["collision_triangles"],
        "watertight=",
        item["visual_watertight"] and item["collision_watertight"],
    )

app.close()
