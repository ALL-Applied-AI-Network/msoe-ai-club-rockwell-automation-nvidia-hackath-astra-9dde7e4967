"""Generate the combined six-class, CSI-style synthetic training dataset.

Classes: M4/M8 screws, round/long pills, and AA/AAA batteries.  Colours are
appearance variants, not semantic classes.  The output intentionally has no
video files.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--scenes", type=int, default=1500)
parser.add_argument("--seed", type=int, default=20261030)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--rt-subframes", type=int, default=4)
parser.add_argument("--max-count", type=int, default=3)
parser.add_argument("--skip-sensor-model", action="store_true")
args = parser.parse_args()

if args.scenes <= 0:
    parser.error("--scenes must be positive")
if not 1 <= args.max_count <= 3:
    parser.error("--max-count must be 1..3")

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

import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, Semantics, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ASSET_ROOT = Path("/workspace/imports/new_pill_battery_assets_candidate_20261029")
SCREW_ROOT = Path("/workspace/imports/screw_bin_sort_candidate_20261027")
PROJECT = "/workspace/fanuc_er4ia_sim"
FINGER = "/workspace/imports/custom_finger_candidate_20261026/integration"
for path in (str(ASSET_ROOT), PROJECT, FINGER):
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


NEW_MANIFEST = json.loads(
    (ASSET_ROOT / "inspection/object_asset_manifest.json").read_text(encoding="utf-8")
)
SCREW_REPORT = json.loads(
    (SCREW_ROOT / "inspection/asset_report.json").read_text(encoding="utf-8")
)["assets"]

CLASSES = (
    "m4_screw",
    "m8_screw",
    "round_pill",
    "long_pill",
    "aa_battery",
    "aaa_battery",
)
FAMILIES = {
    "screw": ("m4_screw", "m8_screw"),
    "pill": ("round_pill", "long_pill"),
    "battery": ("aa_battery", "aaa_battery"),
}
COLORS = {
    "m4_screw": ("black_steel_a", "black_steel_b", "black_steel_c"),
    "m8_screw": ("black_steel_a", "black_steel_b", "black_steel_c"),
    "round_pill": ("red", "yellow", "blue", "black"),
    "long_pill": ("red", "yellow", "blue", "black"),
    "aa_battery": ("battery_black", "battery_blue", "battery_copper", "battery_silver"),
    "aaa_battery": ("battery_black", "battery_blue", "battery_copper", "battery_silver"),
}
MATERIAL_SPECS = {
    "red": ((0.72, 0.025, 0.018), 0.0, 0.40),
    "yellow": ((0.92, 0.64, 0.025), 0.0, 0.42),
    "blue": ((0.025, 0.14, 0.68), 0.0, 0.40),
    "black": ((0.025, 0.028, 0.034), 0.05, 0.35),
    "black_steel_a": ((0.045, 0.048, 0.054), 0.84, 0.25),
    "black_steel_b": ((0.075, 0.080, 0.090), 0.88, 0.34),
    "black_steel_c": ((0.022, 0.025, 0.030), 0.75, 0.21),
    "battery_black": ((0.035, 0.040, 0.048), 0.12, 0.31),
    "battery_blue": ((0.025, 0.12, 0.52), 0.08, 0.34),
    "battery_copper": ((0.48, 0.16, 0.055), 0.38, 0.29),
    "battery_silver": ((0.44, 0.47, 0.50), 0.72, 0.25),
    "terminal_positive": ((0.52, 0.55, 0.58), 0.92, 0.20),
    "terminal_negative": ((0.34, 0.36, 0.39), 0.88, 0.24),
    "bin_red": ((0.72, 0.035, 0.025), 0.0, 0.42),
    "bin_yellow": ((0.95, 0.72, 0.06), 0.0, 0.42),
    "bin_green": ((0.04, 0.52, 0.24), 0.0, 0.42),
    "bin_blue": ((0.08, 0.30, 0.78), 0.0, 0.42),
    "seam": ((0.48, 0.49, 0.50), 0.0, 0.72),
}


def make_material(stage, name, rgb, metallic, roughness):
    path = f"/World/Looks/SixClass_{name}"
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def bind(prim, look):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        look, bindingStrength=UsdShade.Tokens.strongerThanDescendants
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


def add_terminal_visuals(stage, parent_path, dimensions_mm, materials):
    length = float(dimensions_mm[0]) * 0.001
    diameter = float(sum(dimensions_mm[1:]) / 2.0) * 0.001
    positive_height = min(0.0008, max(0.00035, diameter * 0.045))
    negative_height = min(0.00045, max(0.00020, diameter * 0.025))
    terminals = UsdGeom.Xform.Define(stage, parent_path + "/TerminalVisuals")
    positive = UsdGeom.Cylinder.Define(stage, parent_path + "/TerminalVisuals/Positive")
    positive.CreateAxisAttr(UsdGeom.Tokens.x)
    positive.CreateRadiusAttr(diameter * 0.18)
    positive.CreateHeightAttr(positive_height)
    UsdGeom.Xformable(positive).AddTranslateOp().Set(
        Gf.Vec3d(length / 2.0 - positive_height / 2.0, 0.0, 0.0)
    )
    bind(positive.GetPrim(), materials["terminal_positive"])
    negative = UsdGeom.Cylinder.Define(stage, parent_path + "/TerminalVisuals/Negative")
    negative.CreateAxisAttr(UsdGeom.Tokens.x)
    negative.CreateRadiusAttr(diameter * 0.42)
    negative.CreateHeightAttr(negative_height)
    UsdGeom.Xformable(negative).AddTranslateOp().Set(
        Gf.Vec3d(-length / 2.0 + negative_height / 2.0, 0.0, 0.0)
    )
    bind(negative.GetPrim(), materials["terminal_negative"])


def class_spec(class_name):
    if class_name in ("m4_screw", "m8_screw"):
        key = class_name.split("_")[0]
        row = SCREW_REPORT[key + "_visual"]
        bounds = row["bounds_source_units"]
        source_dims = [float(value) for value in bounds["size"]]
        vertical_mm = source_dims[0]
        footprint_mm = [source_dims[2], source_dims[1]]
        return {
            "asset": SCREW_ROOT / f"converted/{key}_visual.usdc",
            "center_mm": bounds["center"],
            "dimensions_mm": [source_dims[2], source_dims[1], source_dims[0]],
            "vertical_m": vertical_mm * 0.001,
            "footprint_m": [value * 0.001 for value in footprint_mm],
            "rotate_y": 90.0,
            "battery": False,
        }
    row = NEW_MANIFEST["assets"][class_name]
    dims = [float(value) for value in row["dimensions_mm"]]
    return {
        "asset": ASSET_ROOT / row["visual_usdc"],
        "center_mm": row["source_bounds_mm"]["center"],
        "dimensions_mm": dims,
        "vertical_m": dims[2] * 0.001,
        "footprint_m": [dims[0] * 0.001, dims[1] * 0.001],
        "rotate_y": None,
        "battery": class_name.endswith("_battery"),
    }


CLASS_SPECS = {name: class_spec(name) for name in CLASSES}


def add_object_slot(stage, class_name, slot_index, materials):
    spec = CLASS_SPECS[class_name]
    path = f"/World/DatasetObjects/{class_name}_{slot_index:02d}"
    root = UsdGeom.Xform.Define(stage, path)
    add_semantic_class(root.GetPrim(), class_name)
    body = add_reference_visual(
        stage,
        path,
        spec["asset"],
        spec["center_mm"],
        rotate_y=spec["rotate_y"],
    )
    if spec["battery"]:
        add_terminal_visuals(stage, path, spec["dimensions_mm"], materials)
    UsdGeom.Imageable(root.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    footprint = spec["footprint_m"]
    radius = 0.5 * math.hypot(footprint[0], footprint[1])
    return {
        "root": root.GetPrim(),
        "body": body,
        "radius_m": radius,
        "vertical_m": spec["vertical_m"],
        "dimensions_mm": spec["dimensions_mm"],
    }


def add_bin_visual(stage, number, xy, look):
    path = f"/World/DatasetBin{number}"
    root = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(root).AddTranslateOp().Set(
        Gf.Vec3d(float(xy[0]), float(xy[1]), float(C.TABLE_TOP_Z + C.SURFACE_DZ))
    )
    mm = UsdGeom.Xform.Define(stage, path + "/Millimetres")
    UsdGeom.Xformable(mm).AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))
    mm.GetPrim().GetReferences().AddReference(str(SCREW_ROOT / "converted/bin.usdc"))
    expand_instances(mm.GetPrim())
    bind(root.GetPrim(), look)


def add_table_marks(stage, look):
    surface = float(C.TABLE_TOP_Z + C.SURFACE_DZ)
    marks = (
        ("SeamA", (0.46, -0.012, 0.00012), (0.22, 0.07, surface + 0.00006)),
        ("SeamB", (0.0012, 0.55, 0.00010), (0.49, 0.00, surface + 0.00005)),
        ("ScuffA", (0.11, 0.0010, 0.00008), (0.43, -0.19, surface + 0.00004)),
        ("ScuffB", (0.08, 0.0008, 0.00008), (0.36, 0.25, surface + 0.00004)),
    )
    for name, scale, position in marks:
        cube = UsdGeom.Cube.Define(stage, "/World/BackgroundMarks/" + name)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        bind(cube.GetPrim(), look)


def set_pose(slot, xyz, yaw_deg, roll_deg):
    xf = UsdGeom.Xformable(slot["root"])
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    xf.AddRotateZOp().Set(float(yaw_deg))
    if abs(roll_deg) > 1e-6:
        xf.AddRotateXOp().Set(float(roll_deg))


def set_visible(prim, visible):
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
        UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    )


def sample_counts(rng):
    while True:
        counts = {name: rng.randint(0, args.max_count) for name in CLASSES}
        total = sum(counts.values())
        family_present = all(sum(counts[name] for name in pair) > 0 for pair in FAMILIES.values())
        if 5 <= total <= 15 and family_present:
            return counts


def sample_layout(rng, counts, slots):
    placed = []
    result = []
    for class_name in CLASSES:
        for slot_index in range(counts[class_name]):
            radius = slots[class_name][slot_index]["radius_m"]
            for _ in range(3000):
                x = rng.uniform(0.305, 0.585)
                y = rng.uniform(-0.325, 0.325)
                if all(
                    math.hypot(x - px, y - py) >= radius + pradius + 0.006
                    for px, py, pradius in placed
                ):
                    placed.append((x, y, radius))
                    result.append((class_name, slot_index, x, y, rng.uniform(-180.0, 180.0)))
                    break
            else:
                raise RuntimeError(f"Could not place counts={counts}")
    rng.shuffle(result)
    return result


def get_xform_value(prim, name, default):
    attr = prim.GetAttribute(name)
    value = attr.Get() if attr and attr.IsValid() else None
    return tuple(float(x) for x in value) if value is not None else default


def setup_lighting(stage):
    result = {}
    for path in ("/World/KeyLight", "/World/FillLight", "/World/RimLight"):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        result[path] = {
            "prim": prim,
            "translate": get_xform_value(prim, "xformOp:translate", (0.0, 0.0, 2.0)),
            "rotate": get_xform_value(prim, "xformOp:rotateXYZ", (160.0, 0.0, 0.0)),
            "intensity": float(UsdLux.LightAPI(prim).GetIntensityAttr().Get()),
        }
    return result


def randomize_lighting(stage, lights, rng):
    profile = rng.choices(
        ("csi_hard_multi", "csi_mixed", "bright_control"),
        weights=(0.68, 0.24, 0.08),
        k=1,
    )[0]
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
    if profile == "csi_hard_multi":
        dome_intensity = rng.uniform(350.0, 650.0)
        size_range = (0.025, 0.16)
        intensity_range = (0.90, 1.80)
        jitter_xy = 0.10
    elif profile == "csi_mixed":
        dome_intensity = rng.uniform(550.0, 850.0)
        size_range = (0.08, 0.30)
        intensity_range = (0.85, 1.45)
        jitter_xy = 0.07
    else:
        dome_intensity = rng.uniform(850.0, 1200.0)
        size_range = (0.30, 0.85)
        intensity_range = (0.85, 1.25)
        jitter_xy = 0.05
    dome.GetIntensityAttr().Set(dome_intensity)

    states = []
    for light_index, (path, data) in enumerate(sorted(lights.items())):
        prim = data["prim"]
        rect = UsdLux.RectLight(prim)
        size = rng.uniform(*size_range)
        rect.GetWidthAttr().Set(size)
        rect.GetHeightAttr().Set(size * rng.uniform(0.75, 1.25))
        scale = rng.uniform(*intensity_range)
        UsdLux.LightAPI(prim).GetIntensityAttr().Set(data["intensity"] * scale)
        warm = rng.uniform(-0.06, 0.06)
        UsdLux.LightAPI(prim).GetColorAttr().Set(
            Gf.Vec3f(1.0, max(0.82, 0.96 - warm), max(0.78, 0.91 - 1.5 * warm))
        )
        translate = (
            data["translate"][0] + rng.uniform(-jitter_xy, jitter_xy),
            data["translate"][1] + rng.uniform(-jitter_xy, jitter_xy),
            data["translate"][2] + rng.uniform(-0.08, 0.12),
        )
        rotate = (
            data["rotate"][0] + rng.uniform(-10.0, 10.0),
            data["rotate"][1] + rng.uniform(-7.0, 7.0),
            data["rotate"][2] + rng.uniform(-18.0, 18.0),
        )
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*translate))
        prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(*rotate))
        states.append(
            {"path": path, "size_m": size, "intensity_scale": scale, "position": list(translate)}
        )
    return {
        "profile": profile,
        "dome_intensity": dome_intensity,
        "rect_lights": states,
    }


def sensor_parameters(rng, lighting_profile):
    hard = lighting_profile == "csi_hard_multi"
    return {
        "exposure": rng.uniform(0.95, 1.35 if hard else 1.25),
        "gamma": rng.uniform(0.82, 1.05),
        "blue_gain": rng.uniform(0.92, 1.05),
        "green_gain": rng.uniform(0.94, 1.06),
        "red_gain": rng.uniform(0.95, 1.09),
        "gaussian_noise_sigma": rng.uniform(2.5, 6.5 if hard else 5.0),
        "blur_sigma": rng.choice((0.0, 0.0, 0.35, 0.55, 0.75)),
        "vignette": rng.uniform(0.01, 0.08),
    }


def apply_sensor_model(path, params, seed):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read generated RGB {path}")
    local = np.random.default_rng(seed)
    data = image.astype(np.float32) / 255.0
    data *= float(params["exposure"])
    data = np.power(np.clip(data, 0.0, 1.0), float(params["gamma"]))
    data[:, :, 0] *= float(params["blue_gain"])
    data[:, :, 1] *= float(params["green_gain"])
    data[:, :, 2] *= float(params["red_gain"])
    height, width = data.shape[:2]
    yy, xx = np.ogrid[-1.0:1.0:height * 1j, -1.0:1.0:width * 1j]
    vignette = 1.0 - float(params["vignette"]) * np.clip(xx * xx + yy * yy, 0.0, 1.0)
    data *= vignette[:, :, None]
    sigma = float(params["blur_sigma"])
    if sigma > 0.0:
        data = cv2.GaussianBlur(data, (0, 0), sigmaX=sigma, sigmaY=sigma)
    noise = local.normal(0.0, float(params["gaussian_noise_sigma"]) / 255.0, data.shape)
    data = np.clip(data + noise.astype(np.float32), 0.0, 1.0)
    result = np.round(data * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write sensor-model RGB {path}")


def split_for_frame(index, total):
    train_end = int(total * 0.80)
    validation_end = int(total * 0.90)
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage, with_objects=False)
finger_config = custom_finger_physics_overlay.apply(
    stage,
    left_rigid_asset=FINGER + "/assets/custom_finger_left_rigid.usda",
    right_rigid_asset=FINGER + "/assets/custom_finger_right_rigid.usda",
    left_silicone_asset=FINGER + "/assets/custom_finger_left_silicone.usda",
    right_silicone_asset=FINGER + "/assets/custom_finger_right_silicone.usda",
    robot_prim=C.ROBOT_PRIM,
    flange_path=cell.FLANGE,
)
materials = {name: make_material(stage, name, *spec) for name, spec in MATERIAL_SPECS.items()}
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
UsdGeom.Xform.Define(stage, "/World/BackgroundMarks")
add_table_marks(stage, materials["seam"])
UsdGeom.Xform.Define(stage, "/World/DatasetObjects")
slots = {
    class_name: [add_object_slot(stage, class_name, index, materials) for index in range(3)]
    for class_name in CLASSES
}

cell.enable_ccd(world)
world.reset()
arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia_six_class_dataset")
arm.initialize()
joint_names = list(arm.dof_names)
positions = arm.get_joint_positions()
for joint, degrees in C.CAPTURE_POSE_DEG.items():
    positions[joint_names.index(joint)] = np.deg2rad(degrees)
for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
    positions[joint_names.index(joint)] = value
arm.set_joint_positions(positions)
arm.apply_action(ArticulationAction(joint_positions=positions))
for _ in range(24):
    world.step(render=True)

lights = setup_lighting(stage)
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
rows = []
split_files = {name: [] for name in ("train", "validation", "test")}
for frame_index in range(args.scenes):
    counts = sample_counts(rng)
    layout = sample_layout(rng, counts, slots)
    active = {(class_name, slot_index) for class_name, slot_index, *_ in layout}
    for class_name in CLASSES:
        for slot_index, slot in enumerate(slots[class_name]):
            set_visible(slot["root"], (class_name, slot_index) in active)

    objects = []
    for class_name, slot_index, x, y, yaw in layout:
        slot = slots[class_name][slot_index]
        color = rng.choice(COLORS[class_name])
        roll = rng.uniform(-180.0, 180.0) if class_name.endswith("_battery") else 0.0
        z = surface_z + slot["vertical_m"] / 2.0 + 0.00015
        set_pose(slot, (x, y, z), yaw, roll)
        bind(slot["body"], materials[color])
        objects.append(
            {
                "name": slot["root"].GetName(),
                "semantic_class": class_name,
                "family": next(name for name, pair in FAMILIES.items() if class_name in pair),
                "color": color,
                "position_m": [x, y, z],
                "yaw_deg": yaw,
                "roll_deg": roll,
                "dimensions_mm": slot["dimensions_mm"],
            }
        )

    lighting = randomize_lighting(stage, lights, rng)
    sensor = sensor_parameters(rng, lighting["profile"])
    rep.orchestrator.step(
        rt_subframes=args.rt_subframes,
        pause_timeline=True,
        delta_time=0.0,
        wait_for_render=True,
    )
    split = split_for_frame(frame_index, args.scenes)
    split_files[split].append(f"rgb_{frame_index:06d}.png")
    row = {
        "frame_index": frame_index,
        "split": split,
        "rgb_file": f"rgb_{frame_index:06d}.png",
        "bbox_file": f"bounding_box_2d_tight_{frame_index:06d}.npy",
        "semantic_segmentation_file": f"semantic_segmentation_{frame_index:06d}.png",
        "seed": args.seed,
        "object_counts": counts,
        "objects": objects,
        "lighting": lighting,
        "sensor_model": sensor,
    }
    rows.append(row)
    print(
        f"[dataset] frame {frame_index + 1}/{args.scenes} split={split} "
        + " ".join(f"{name}={counts[name]}" for name in CLASSES)
        + f" light={lighting['profile']}",
        flush=True,
    )

rep.orchestrator.wait_until_complete()
writer.detach()

if not args.skip_sensor_model:
    for index, row in enumerate(rows):
        apply_sensor_model(
            out_dir / row["rgb_file"],
            row["sensor_model"],
            args.seed * 1000003 + index,
        )
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"[sensor] {index + 1}/{len(rows)}", flush=True)

with (out_dir / "scene_metadata.jsonl").open("w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
for split, names in split_files.items():
    (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

manifest = {
    "status": "GENERATED",
    "task": "combined_six_class_detection",
    "semantic_classes": list(CLASSES),
    "families": {name: list(pair) for name, pair in FAMILIES.items()},
    "color_is_not_a_class": True,
    "old_pill_assets_reused": False,
    "scenes": args.scenes,
    "seed": args.seed,
    "count_per_class_per_scene": [0, args.max_count],
    "every_scene_contains_each_family": True,
    "resolution": [args.width, args.height],
    "splits": {name: len(files) for name, files in split_files.items()},
    "shadow_profiles": ["csi_hard_multi", "csi_mixed", "bright_control"],
    "sensor_model_applied": not args.skip_sensor_model,
    "real_reference": (
        "/workspace host archive corresponds to local "
        "/home/anik/Documents/resutls/real_images_20261029"
    ),
    "include_four_bin_distractors": True,
    "table_surface_world_z_m": surface_z,
    "finger": "custom finger",
    "finger_physics": finger_config,
    "video_generated": False,
    "outputs": {
        "rgb": "rgb_*.png",
        "tight_2d_boxes": "bounding_box_2d_tight_*.npy",
        "box_labels": "bounding_box_2d_tight_labels_*.json",
        "semantic_segmentation": "semantic_segmentation_*.png",
        "semantic_labels": "semantic_segmentation_labels_*.json",
        "scene_metadata": "scene_metadata.jsonl",
        "split_lists": ["train.txt", "validation.txt", "test.txt"],
    },
}
with (out_dir / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)

print(f"[dataset] complete: {args.scenes} scenes in {out_dir}")
app.close()
