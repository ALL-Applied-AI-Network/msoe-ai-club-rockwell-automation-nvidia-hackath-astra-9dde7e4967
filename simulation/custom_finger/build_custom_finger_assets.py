"""Build mount-centred custom finger wrappers with separate visual/collision trees."""

import argparse
import hashlib
import json
import os

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("--visual", required=True)
parser.add_argument("--collision", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom


MOUNT_TO_TIP_M = 0.0687643899917603
OPEN_INNER_GAP_M = 0.0164
CLOSED_INNER_GAP_M = 0.0044
MOUNT_OFFSET_M = 0.0052
SOURCE_MOUNTS_MM = {
    "left": (0.0, 0.0, 28.25),
    "right": (0.0, 0.0, -1.75),
}
COMPONENTS = {
    "left": {
        "rigid": ("LEFT_TPU_BASE", "LEFT_TPU_BASE_COLLISION"),
        "silicone": ("LEFT_SILICONE_PAD", "LEFT_SILICONE_PAD_COLLISION"),
    },
    "right": {
        "rigid": ("RIGHT_TPU_BASE", "RIGHT_TPU_BASE_COLLISION"),
        "silicone": ("RIGHT_SILICONE_PAD", "RIGHT_SILICONE_PAD_COLLISION"),
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_path(asset_kind, component):
    return f"/custom_finger_{asset_kind}/custom_finger_{asset_kind}/{component}"


def add_branch(stage, root, branch_name, asset_path, prim_path, mount_source_mm):
    branch = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild(branch_name))
    oriented = UsdGeom.Xform.Define(
        stage, branch.GetPath().AppendChild("OrientedCAD")
    )
    # The CAD converter retained SolidWorks Y-up geometry in a Z-up stage.
    # Rotate -90 degrees about X: source separation Z -> robot closing Y,
    # and source physical vertical Y -> robot vertical -Z.
    oriented.AddRotateXOp().Set(-90.0)
    oriented.AddScaleOp().Set(Gf.Vec3d(0.001))
    source_frame = UsdGeom.Xform.Define(
        stage, oriented.GetPath().AppendChild("MountFrame")
    )
    source_frame.AddTranslateOp().Set(-Gf.Vec3d(*mount_source_mm))
    source = UsdGeom.Xform.Define(
        stage, source_frame.GetPath().AppendChild("SourceComponent")
    )
    wrapper_dir = os.path.dirname(stage.GetRootLayer().realPath)
    reference_path = os.path.relpath(asset_path, wrapper_dir).replace(os.sep, "/")
    if not reference_path.startswith("."):
        reference_path = "./" + reference_path
    source.GetPrim().GetReferences().AddReference(reference_path, prim_path)
    return branch


def bounds(stage, prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return {
        "min_m": [float(value) for value in extent.GetMin()],
        "max_m": [float(value) for value in extent.GetMax()],
        "size_m": [float(value) for value in extent.GetSize()],
    }


def build(visual_path, collision_path, out_dir, side, role, names):
    output = os.path.join(out_dir, f"custom_finger_{side}_{role}.usda")
    stage = Usd.Stage.CreateNew(output)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(
        stage, f"/FingerV2{side.capitalize()}{role.capitalize()}"
    )
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("integration:revision", "finger_v02_candidate")
    root.GetPrim().SetCustomDataByKey("integration:role", role)
    root.GetPrim().SetCustomDataByKey("integration:side", side)
    root.GetPrim().SetCustomDataByKey("integration:frame", "mount_screw_candidate")
    mount = SOURCE_MOUNTS_MM[side]
    visual_branch = add_branch(
        stage,
        root,
        "Visual",
        visual_path,
        component_path("visual", names[0]),
        mount,
    )
    collision_branch = add_branch(
        stage,
        root,
        "Collision",
        collision_path,
        component_path("collision", names[1]),
        mount,
    )
    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(output)
    check_root = check.GetDefaultPrim()
    return output, {
        "visual": bounds(check, check_root.GetChild("Visual")),
        "collision": bounds(check, check_root.GetChild("Collision")),
        "overall": bounds(check, check_root),
    }


visual_path = os.path.abspath(args.visual)
collision_path = os.path.abspath(args.collision)
out_dir = os.path.abspath(args.out)
os.makedirs(out_dir, exist_ok=True)

visual_stage = Usd.Stage.Open(visual_path)
collision_stage = Usd.Stage.Open(collision_path)
if not visual_stage or not collision_stage:
    raise RuntimeError("Could not open one or both converted source assets")

manifest = {
    "finger_id": "finger_v02_candidate",
    "source_visual": visual_path,
    "source_visual_sha256": sha256(visual_path),
    "source_collision": collision_path,
    "source_collision_sha256": sha256(collision_path),
    "mount_frame_assumption": {
        "description": "top mounting-screw centre inferred from symmetric CAD geometry",
        "source_mounts_mm": SOURCE_MOUNTS_MM,
        "axis_remap": "rotate -90 degrees about X",
    },
    "calibration": {
        "open_inner_gap_m": OPEN_INNER_GAP_M,
        "closed_inner_gap_m": CLOSED_INNER_GAP_M,
        "actuator_travel_per_jaw_m": 0.006,
        "mount_offset_per_side_m": MOUNT_OFFSET_M,
        "mount_to_tip_m": MOUNT_TO_TIP_M,
        "silicone_compression_limit_per_side_m": 0.001,
    },
    "material_assumption": {
        "core": "PLA per project owner; CAD labels it TPU",
        "pad": "printed silicone",
    },
    "assets": {},
}

for side, roles in COMPONENTS.items():
    manifest["assets"][side] = {}
    for role, names in roles.items():
        for asset_kind, asset_stage, component in (
            ("visual", visual_stage, names[0]),
            ("collision", collision_stage, names[1]),
        ):
            path = component_path(asset_kind, component)
            if not asset_stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Missing source component {path}")
        output, asset_bounds = build(
            visual_path, collision_path, out_dir, side, role, names
        )
        manifest["assets"][side][role] = {
            "path": output,
            "source_components": {
                "visual": names[0],
                "collision": names[1],
            },
            "bounds": asset_bounds,
        }
        print(
            "[custom-finger-asset]",
            side,
            role,
            "visual_size_m=",
            tuple(round(value, 6) for value in asset_bounds["visual"]["size_m"]),
            "collision_size_m=",
            tuple(round(value, 6) for value in asset_bounds["collision"]["size_m"]),
            flush=True,
        )

manifest_path = os.path.join(out_dir, "custom_finger_asset_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
print("[custom-finger-asset] manifest", manifest_path, flush=True)
app.close()
