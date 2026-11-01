"""Sweep the custom finger core/silicone fingers through their range with 3 cameras."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--run", required=True)
parser.add_argument("--left-rigid", required=True)
parser.add_argument("--right-rigid", required=True)
parser.add_argument("--left-silicone", required=True)
parser.add_argument("--right-silicone", required=True)
parser.add_argument("--overlay-module", required=True)
args = parser.parse_args()
run_dir = Path(args.run).resolve()
for camera_name in ("table", "side", "wrist"):
    frame_dir = run_dir / f"{camera_name}_frames"
    if frame_dir.exists() and any(frame_dir.glob("*.png")):
        raise SystemExit(f"Refusing to overwrite frames in {frame_dir}")

from isaacsim import SimulationApp


app = SimulationApp({"headless": True})

import cv2
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera


sys.path.insert(0, os.path.abspath(args.project))
sys.path.insert(0, os.path.dirname(os.path.abspath(args.overlay_module)))
import cell
import custom_finger_physics_overlay as finger_physics_overlay
import scene_config as C


# Keep the project defaults intact and calibrate only this isolated process.
previous_finger_length = C.FINGER_LENGTH
C.FINGER_LENGTH = finger_physics_overlay.MOUNT_TO_TIP_M
C.TOOL_LENGTH += C.FINGER_LENGTH - previous_finger_length
C.CUBE_JAW_HALF_TARGET = 0.0025

world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
physics_config = finger_physics_overlay.apply(
    stage,
    left_rigid_asset=os.path.abspath(args.left_rigid),
    right_rigid_asset=os.path.abspath(args.right_rigid),
    left_silicone_asset=os.path.abspath(args.left_silicone),
    right_silicone_asset=os.path.abspath(args.right_silicone),
    robot_prim=C.ROBOT_PRIM,
    flange_path=cell.FLANGE,
)

cameras = {
    "table": Camera(cell.TABLE_CAM_PATH, resolution=C.TABLE_CAM["resolution"]),
    "side": Camera(cell.SIDE_CAM_PATH, resolution=C.SIDE_CAM["resolution"]),
    "wrist": Camera(cell.WRIST_CAM_PATH, resolution=C.WRIST_CAM["resolution"]),
}
for camera_name in cameras:
    (run_dir / f"{camera_name}_frames").mkdir(parents=True, exist_ok=True)

cell.enable_ccd(world)
world.reset()
for camera in cameras.values():
    camera.initialize()

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia_finger_clearance")
arm.initialize()
names = list(arm.dof_names)
required = [
    *(f"J{index}" for index in range(1, 7)),
    "jaw_l",
    "jaw_r",
    "silicone_l_compression",
    "silicone_r_compression",
]
missing = [name for name in required if name not in names]
if missing:
    raise RuntimeError(f"Missing expected articulation DOFs: {missing}; found {names}")
indices = {name: names.index(name) for name in names}
arm_indices = [indices[f"J{index}"] for index in range(1, 7)]
controlled_indices = arm_indices + [indices["jaw_l"], indices["jaw_r"]]

q_arm = np.deg2rad([C.HOME_POSE_DEG[f"J{index}"] for index in range(1, 7)])
initial = arm.get_joint_positions().copy()
initial[arm_indices] = q_arm
initial[indices["jaw_l"]] = 0.0
initial[indices["jaw_r"]] = 0.0
initial[indices["silicone_l_compression"]] = 0.0
initial[indices["silicone_r_compression"]] = 0.0
arm.set_joint_positions(initial)


def command(jaw_opening):
    jaw_targets = cell.jaw_targets(jaw_opening)
    values = np.concatenate(
        [q_arm, [jaw_targets["jaw_l"], jaw_targets["jaw_r"]]]
    )
    arm.apply_action(
        ArticulationAction(
            joint_positions=values,
            joint_indices=np.asarray(controlled_indices, dtype=np.int32),
        )
    )


# RTX camera annotators need several rendered frames after initialization.
# Hold the articulation at its safe open pose and discard these warm-up frames.
for _ in range(30):
    command(C.JAW_OPEN)
    world.step(render=True)

segments = [
    ("open_hold", C.JAW_OPEN, C.JAW_OPEN, 30),
    ("close", C.JAW_OPEN, C.JAW_CLOSED, 60),
    ("closed_hold", C.JAW_CLOSED, C.JAW_CLOSED, 30),
    ("open", C.JAW_CLOSED, C.JAW_OPEN, 60),
    ("final_hold", C.JAW_OPEN, C.JAW_OPEN, 30),
]

samples = []
frame = 0
for phase, start, end, count in segments:
    for phase_frame in range(count):
        fraction = (phase_frame + 1) / count
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        requested = start + (end - start) * blend
        command(requested)
        for _ in range(2):
            world.step(render=True)

        for camera_name, camera in cameras.items():
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError(f"{camera_name} camera returned no image")
            cv2.imwrite(
                str(run_dir / f"{camera_name}_frames/f{frame:05d}.png"),
                cv2.cvtColor(
                    rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                ),
            )

        measured = arm.get_joint_positions()
        samples.append(
            {
                "frame": frame,
                "phase": phase,
                "requested_jaw_opening_m": float(requested),
                "jaw_l_m": float(measured[indices["jaw_l"]]),
                "jaw_r_m": float(measured[indices["jaw_r"]]),
                "silicone_l_m": float(
                    measured[indices["silicone_l_compression"]]
                ),
                "silicone_r_m": float(
                    measured[indices["silicone_r_compression"]]
                ),
                "configured_inner_gap_m": float(
                    finger_physics_overlay.CLOSED_INNER_GAP_M + 2.0 * requested
                ),
            }
        )
        frame += 1

max_unloaded_compression = max(
    max(abs(sample["silicone_l_m"]), abs(sample["silicone_r_m"]))
    for sample in samples
)
report = {
    "test": "custom_finger_physics_clearance",
    "task_complete": bool(max_unloaded_compression < 0.0001),
    "frames_per_camera": frame,
    "fps": 30,
    "duration_s": frame / 30,
    "dof_names": names,
    "compression_limit_per_side_m": 0.001,
    "configured_open_inner_gap_m": finger_physics_overlay.OPEN_INNER_GAP_M,
    "configured_closed_inner_gap_m": finger_physics_overlay.CLOSED_INNER_GAP_M,
    "cube_jaw_half_target_m": C.CUBE_JAW_HALF_TARGET,
    "max_unloaded_compression_m": float(max_unloaded_compression),
    "physics_config": physics_config,
    "tool_length_m": C.TOOL_LENGTH,
    "samples": samples,
}
with open(run_dir / "clearance_report.json", "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)

print("[finger-clearance] frames_per_camera", frame)
print(
    "[finger-clearance] max_unloaded_compression_mm",
    round(max_unloaded_compression * 1000.0, 6),
)
print("[finger-clearance] task_complete", report["task_complete"])
app.close()
