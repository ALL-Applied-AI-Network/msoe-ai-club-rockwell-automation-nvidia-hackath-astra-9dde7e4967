"""Run policy 2's production perception path on the exact headless scene."""

from collections import Counter
import json
from pathlib import Path
import sys

import cv2


POLICY = Path("/home/yt-astraf0a5/workspace/physical_ai_submission_policy4")
RUN = Path("/home/yt-astraf0a5/workspace/fanuc_er4ia_sim/runs/policy2_four_class_scene_01")
sys.path.insert(0, str(POLICY))

from sort4_policy.instruction import EXPECTED_COUNTS, default_task
from sort4_policy.perception import Sort4CosmosClient


instruction = (
    "Sort all M4 screws into the red bin, all M8 screws into the yellow bin, "
    "all red cubes into the green bin, and all blue cubes into the blue bin."
)
bgr = cv2.imread(str(RUN / "world_camera.png"))
if bgr is None:
    raise SystemExit("positive scene image is missing")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
task = default_task(instruction)
client = Sort4CosmosClient()
scene = client.infer(rgb, task)
targets = client.to_targets(scene, task)
counts = Counter(target.sort_class for target in targets)
if dict(counts) != EXPECTED_COUNTS:
    raise SystemExit(f"unexpected target inventory: {dict(counts)}")
report = {
    "status": "PASS",
    "model": scene.get("model"),
    "evidence": scene.get("evidence"),
    "verification": scene.get("verification"),
    "object_crop_pixels": scene.get("object_crop_pixels"),
    "inventory": dict(counts),
    "bins": scene["bins"],
    "targets": [
        {
            "class": target.sort_class,
            "object_xy_mm": target.object_xy_mm,
            "yaw_deg": target.object_yaw_deg,
            "bin_color": target.bin_color,
            "bin_xy_mm": target.bin_xy_mm,
        }
        for target in targets
    ],
}
(RUN / "policy2_perception_validation.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
