"""Generate randomized five-class RGB/annotation scenes in the calibrated cell.

The five semantic classes are shape classes.  Colour is randomized independently
and recorded in scene_metadata.jsonl.  This script intentionally writes no video.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="New or empty output directory")
parser.add_argument("--scenes", type=int, default=500)
parser.add_argument("--seed", type=int, default=20261029)
parser.add_argument("--min-count", type=int, default=1)
parser.add_argument("--max-count", type=int, default=3)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--rt-subframes", type=int, default=4)
parser.add_argument("--without-bins", action="store_true")
args = parser.parse_args()

if args.scenes <= 0:
    parser.error("--scenes must be positive")
if args.min_count < 0 or args.max_count < args.min_count:
    parser.error("invalid --min-count/--max-count")
if args.max_count > 3:
    parser.error("this generator pre-allocates at most three objects per class")

out_dir = Path(args.out).resolve()
if out_dir.exists() and any(out_dir.iterdir()):
    raise RuntimeError(f"Output directory must be empty: {out_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp


app = SimulationApp(
    {
        "headless": True,
        "width": args.width,
        "height": args.height,
        "anti_aliasing": 3,
    }
)

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, Semantics, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ASSET_ROOT = Path("/workspace/imports/five_class_assets_candidate_20261029")
SCREW_ROOT = Path("/workspace/imports/screw_bin_sort_candidate_20261027")
PROJECT = "/workspace/fanuc_er4ia_sim"
FINGER_INTEGRATION = "/workspace/imports/custom_finger_candidate_20261026/integration"
for path in (str(ASSET_ROOT), PROJECT, FINGER_INTEGRATION):
    if path not in sys.path:
        sys.path.insert(0, path)

import scene_config as C


previous_length = C.FINGER_LENGTH
C.FINGER_LENGTH = 0.0687643899917603
C.TOOL_LENGTH += C.FINGER_LENGTH - previous_length
C.SHOW_BINS = False
C.SHOW_FIXTURE = False

import cell
import custom_finger_physics_overlay


CLASSES = ("m4_screw", "m8_screw", "cube", "long_pill", "round_pill")
CLASS_GEOMETRY = {
    "m4_screw": {"size_mm": (7.0, 7.0, 24.0), "radius_m": 0.0130},
    "m8_screw": {"size_mm": (13.0, 13.0, 28.0), "radius_m": 0.0160},
    "cube": {"size_mm": (10.0, 10.0, 10.0), "radius_m": 0.0075},
    "long_pill": {"size_mm": (19.5, 6.5, 6.5), "radius_m": 0.0110},
    "round_pill": {"size_mm": (10.0, 10.0, 5.0), "radius_m": 0.0075},
}
COLORS = {
    "cube": ("red", "blue", "yellow"),
    "long_pill": ("red", "yellow"),
    "round_pill": ("red", "yellow"),
    "m4_screw": ("black_steel_a", "black_steel_b", "black_steel_c"),
    "m8_screw": ("black_steel_a", "black_steel_b", "black_steel_c"),
}
MATERIAL_SPECS = {
    "red": ((0.78, 0.055, 0.035), 0.0, 0.40),
    "blue": ((0.035, 0.18, 0.78), 0.0, 0.42),
    "yellow": ((0.94, 0.68, 0.045), 0.0, 0.43),
    "black_steel_a": ((0.055, 0.060, 0.068), 0.82, 0.27),
    "black_steel_b": ((0.090, 0.095, 0.105), 0.88, 0.34),
    "black_steel_c": ((0.030, 0.034, 0.040), 0.74, 0.22),
    "bin_red": ((0.72, 0.035, 0.025), 0.0, 0.42),
    "bin_yellow": ((0.95, 0.72, 0.06), 0.0, 0.42),
    "bin_green": ((0.04, 0.52, 0.24), 0.0, 0.42),
    "bin_blue": ((0.08, 0.30, 0.78), 0.0, 0.42),
}


def material(stage, name, rgb, metallic, roughness):
    path = f"/World/Looks/Dataset_{name}"
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


def add_semantic_class(prim, class_name):
    instance_name = Tf.MakeValidIdentifier(f"class_{class_name}")
    api = Semantics.SemanticsAPI.Apply(prim, instance_name)
    api.CreateSemanticTypeAttr().Set("class")
    api.CreateSemanticDataAttr().Set(class_name)


def expand_instances(root):
    while True:
        instances = [prim for prim in Usd.PrimRange(root) if prim.IsInstance()]
        if not instances:
            return
        for prim in instances:
            prim.SetInstanceable(False)


def add_reference_visual(stage, parent_path, asset, source_center_mm, rotate_y=None):
    align = UsdGeom.Xform.Define(stage, parent_path + "/Align")
    if rotate_y is not None:
        UsdGeom.Xformable(align).AddRotateYOp().Set(float(rotate_y))
    mm = UsdGeom.Xform.Define(stage, parent_path + "/Align/Millimetres")
    UsdGeom.Xformable(mm).AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))
    visual = UsdGeom.Xform.Define(stage, parent_path + "/Align/Millimetres/Visual")
    UsdGeom.Xformable(visual).AddTranslateOp().Set(
        Gf.Vec3d(*[-float(value) for value in source_center_mm])
    )
    visual.GetPrim().GetReferences().AddReference(str(asset))
    expand_instances(visual.GetPrim())
    for prim in Usd.PrimRange(visual.GetPrim()):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    return visual.GetPrim()


def add_object_slot(stage, class_name, slot_index):
    path = f"/World/DatasetObjects/{class_name}_{slot_index:02d}"
    root = UsdGeom.Xform.Define(stage, path)
    add_semantic_class(root.GetPrim(), class_name)
    if class_name == "cube":
        geom = UsdGeom.Cube.Define(stage, path + "/Geometry")
        geom.CreateSizeAttr(0.010)
    elif class_name == "m4_screw":
        add_reference_visual(
            stage,
            path,
            SCREW_ROOT / "converted/m4_visual.usdc",
            (0.0, 0.0, -12.0),
            rotate_y=90.0,
        )
    elif class_name == "m8_screw":
        add_reference_visual(
            stage,
            path,
            SCREW_ROOT / "converted/m8_visual.usdc",
            (0.0, 0.0, -14.0),
            rotate_y=90.0,
        )
    elif class_name == "long_pill":
        add_reference_visual(
            stage,
            path,
            ASSET_ROOT / "converted/long_pill_visual.usdc",
            (0.0, 0.0, 3.25),
        )
    elif class_name == "round_pill":
        add_reference_visual(
            stage,
            path,
            ASSET_ROOT / "converted/round_pill_visual.usdc",
            (0.0, 0.0, 2.5),
        )
    else:
        raise KeyError(class_name)
    UsdGeom.Imageable(root.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return root.GetPrim()


def add_bin_visual(stage, number, xy, mat):
    # Static visual distractor.  Bins intentionally have no class semantics.
    path = f"/World/DatasetBin{number}"
    root = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(root).AddTranslateOp().Set(
        Gf.Vec3d(float(xy[0]), float(xy[1]), float(C.TABLE_TOP_Z + C.SURFACE_DZ))
    )
    visual = UsdGeom.Xform.Define(stage, path + "/Millimetres")
    UsdGeom.Xformable(visual).AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))
    visual.GetPrim().GetReferences().AddReference(str(SCREW_ROOT / "converted/bin.usdc"))
    expand_instances(visual.GetPrim())
    bind(root.GetPrim(), mat)


def set_pose(prim, xyz, yaw_deg):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    xf.AddRotateZOp().Set(float(yaw_deg))


def set_visible(prim, visible):
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
        UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    )


def sample_layout(rng, counts):
    # This rectangle matches the loose-object half of the CSI work mat and stays
    # clear of the four-bin row at x=0.205 and of the robot base at x=0.
    placed = []
    result = []
    for class_name in CLASSES:
        radius = CLASS_GEOMETRY[class_name]["radius_m"]
        for slot_index in range(counts[class_name]):
            for _ in range(2000):
                x = rng.uniform(0.315, 0.545)
                y = rng.uniform(-0.300, 0.300)
                if all(
                    math.hypot(x - px, y - py) >= radius + pradius + 0.010
                    for px, py, pradius in placed
                ):
                    placed.append((x, y, radius))
                    result.append((class_name, slot_index, x, y, rng.uniform(-180.0, 180.0)))
                    break
            else:
                raise RuntimeError(f"Could not place scene with counts={counts}")
    rng.shuffle(result)
    return result


def center_height(class_name):
    if class_name == "m4_screw":
        return 0.0035
    if class_name == "m8_screw":
        return 0.0065
    if class_name == "cube":
        return 0.0050
    if class_name == "long_pill":
        return 0.00325
    if class_name == "round_pill":
        return 0.00250
    raise KeyError(class_name)


def camera_metadata(stage):
    camera_prim = stage.GetPrimAtPath(cell.TABLE_CAM_PATH)
    schema = UsdGeom.Camera(camera_prim)
    transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
    return {
        "prim_path": cell.TABLE_CAM_PATH,
        "resolution": [args.width, args.height],
        "focal_length_mm": float(schema.GetFocalLengthAttr().Get()),
        "horizontal_aperture_mm": float(schema.GetHorizontalApertureAttr().Get()),
        "vertical_aperture_mm": float(schema.GetVerticalApertureAttr().Get()),
        "camera_to_world": [
            [float(transform[row][column]) for column in range(4)] for row in range(4)
        ],
    }


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage, with_objects=False)
finger_config = custom_finger_physics_overlay.apply(
    stage,
    left_rigid_asset=FINGER_INTEGRATION + "/assets/custom_finger_left_rigid.usda",
    right_rigid_asset=FINGER_INTEGRATION + "/assets/custom_finger_right_rigid.usda",
    left_silicone_asset=FINGER_INTEGRATION + "/assets/custom_finger_left_silicone.usda",
    right_silicone_asset=FINGER_INTEGRATION + "/assets/custom_finger_right_silicone.usda",
    robot_prim=C.ROBOT_PRIM,
    flange_path=cell.FLANGE,
)

materials = {
    name: material(stage, name, *spec) for name, spec in MATERIAL_SPECS.items()
}
if not args.without_bins:
    for number, (xy, mat_name) in enumerate(
        (
            ((0.205, 0.285), "bin_yellow"),
            ((0.205, 0.095), "bin_red"),
            ((0.205, -0.095), "bin_blue"),
            ((0.205, -0.285), "bin_green"),
        ),
        start=1,
    ):
        add_bin_visual(stage, number, xy, materials[mat_name])

UsdGeom.Xform.Define(stage, "/World/DatasetObjects")
slots = {
    class_name: [add_object_slot(stage, class_name, index) for index in range(3)]
    for class_name in CLASSES
}

cell.enable_ccd(world)
world.reset()
arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia_dataset")
arm.initialize()
joint_names = list(arm.dof_names)
positions = arm.get_joint_positions()
for joint, degrees in C.CAPTURE_POSE_DEG.items():
    positions[joint_names.index(joint)] = np.deg2rad(degrees)
for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
    positions[joint_names.index(joint)] = value
arm.set_joint_positions(positions)
arm.apply_action(ArticulationAction(joint_positions=positions))
for _ in range(20):
    world.step(render=True)

render_product = rep.create.render_product(cell.TABLE_CAM_PATH, (args.width, args.height))
writer = rep.writers.get("BasicWriter")
writer.initialize(
    output_dir=str(out_dir),
    semantic_types=["class"],
    rgb=True,
    bounding_box_2d_tight=True,
    semantic_segmentation=True,
    colorize_semantic_segmentation=True,
    frame_padding=6,
)
writer.attach(render_product)

rng = random.Random(args.seed)
surface_z = float(C.TABLE_TOP_Z + C.SURFACE_DZ)
light_paths = ("/World/DomeLight", "/World/KeyLight", "/World/FillLight", "/World/RimLight")
light_bases = {}
for path in light_paths:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        attr = UsdLux.LightAPI(prim).GetIntensityAttr()
        light_bases[path] = float(attr.Get())

metadata_path = out_dir / "scene_metadata.jsonl"
with metadata_path.open("w", encoding="utf-8") as metadata_stream:
    for frame_index in range(args.scenes):
        counts = {name: rng.randint(args.min_count, args.max_count) for name in CLASSES}
        layout = sample_layout(rng, counts)
        active = {(class_name, slot_index) for class_name, slot_index, *_ in layout}
        for class_name in CLASSES:
            for slot_index, prim in enumerate(slots[class_name]):
                set_visible(prim, (class_name, slot_index) in active)

        objects = []
        for class_name, slot_index, x, y, yaw in layout:
            prim = slots[class_name][slot_index]
            color = rng.choice(COLORS[class_name])
            set_pose(prim, (x, y, surface_z + center_height(class_name)), yaw)
            bind(prim, materials[color])
            objects.append(
                {
                    "name": prim.GetName(),
                    "semantic_class": class_name,
                    "color": color,
                    "position_m": [x, y, surface_z + center_height(class_name)],
                    "yaw_deg": yaw,
                    "dimensions_mm": list(CLASS_GEOMETRY[class_name]["size_mm"]),
                }
            )

        light_scale = rng.uniform(0.72, 1.28)
        for path, base in light_bases.items():
            UsdLux.LightAPI(stage.GetPrimAtPath(path)).GetIntensityAttr().Set(base * light_scale)

        rep.orchestrator.step(
            rt_subframes=args.rt_subframes,
            pause_timeline=True,
            delta_time=0.0,
            wait_for_render=True,
        )
        row = {
            "frame_index": frame_index,
            "rgb_file": f"rgb_{frame_index:06d}.png",
            "bbox_file": f"bounding_box_2d_tight_{frame_index:06d}.npy",
            "semantic_segmentation_file": f"semantic_segmentation_{frame_index:06d}.png",
            "seed": args.seed,
            "object_counts": counts,
            "objects": objects,
            "light_scale": light_scale,
        }
        metadata_stream.write(json.dumps(row, sort_keys=True) + "\n")
        metadata_stream.flush()
        print(
            f"[dataset] frame {frame_index + 1}/{args.scenes}: "
            + ", ".join(f"{name}={counts[name]}" for name in CLASSES),
            flush=True,
        )

rep.orchestrator.wait_until_complete()
writer.detach()

manifest = {
    "status": "PASS",
    "task": "five_class_shape_detection",
    "semantic_classes": list(CLASSES),
    "color_is_not_a_class": True,
    "scenes": args.scenes,
    "seed": args.seed,
    "counts_per_class": [args.min_count, args.max_count],
    "resolution": [args.width, args.height],
    "table_surface_world_z_m": surface_z,
    "include_four_bin_distractors": not args.without_bins,
    "camera": camera_metadata(stage),
    "finger": "custom finger",
    "finger_physics": finger_config,
    "outputs": {
        "rgb": "rgb_*.png",
        "tight_2d_boxes": "bounding_box_2d_tight_*.npy",
        "box_labels": "bounding_box_2d_tight_labels_*.json",
        "semantic_segmentation": "semantic_segmentation_*.png",
        "semantic_labels": "semantic_segmentation_labels_*.json",
        "scene_metadata": "scene_metadata.jsonl",
    },
}
with (out_dir / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)

print(f"[dataset] complete: {args.scenes} scenes in {out_dir}")
app.close()
