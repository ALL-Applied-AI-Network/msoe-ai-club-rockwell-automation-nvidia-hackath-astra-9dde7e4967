"""Final read-only consistency/safety audit for Astra policy 2."""

from collections import Counter
import json
from pathlib import Path
import sys


POLICY = Path("/home/yt-astraf0a5/workspace/physical_ai_submission_policy4")
RUN = Path("/home/yt-astraf0a5/workspace/fanuc_er4ia_sim/runs/policy2_four_class_scene_01")
MOCK = POLICY / "runtime_logs/policy2_20261028T002717Z_38224"
sys.path.insert(0, str(POLICY))

from sort4_policy.trajectory import safe_pose


report = json.loads((RUN / "policy2_perception_validation.json").read_text())
expected_counts = {"m4_screw": 3, "m8_screw": 3, "red_cube": 2, "blue_cube": 2}
counts = Counter(target["class"] for target in report["targets"])
if dict(counts) != expected_counts:
    raise SystemExit(f"inventory failed: {dict(counts)}")

grasp_z = {"m4_screw": -327, "m8_screw": -327, "red_cube": -325, "blue_cube": -325}
for target in report["targets"]:
    x, y = target["object_xy_mm"]
    bx, by = target["bin_xy_mm"]
    yaw = target["yaw_deg"]
    for pose in (
        [x, y, -245, -180, 0, yaw, 0],
        [x, y, grasp_z[target["class"]], -180, 0, yaw, 0],
        [x, y, -225, -180, 0, yaw, 1],
        [bx, by, -225, -180, 0, 0, 1],
        [bx, by, -270, -180, 0, 0, 1],
    ):
        safe_pose(pose)

expected_bins = {"red": (205, 285), "yellow": (205, 95), "green": (205, -95), "blue": (205, -285)}
observed_bins = {}
for target in report["targets"]:
    observed_bins.setdefault(target["bin_color"], target["bin_xy_mm"])
bin_errors = {
    color: ((xy[0] - expected_bins[color][0]) ** 2 + (xy[1] - expected_bins[color][1]) ** 2) ** 0.5
    for color, xy in observed_bins.items()
}
if max(bin_errors.values()) > 3.0:
    raise SystemExit(f"bin calibration error too large: {bin_errors}")

videos = sorted(MOCK.glob("mock-t*.mp4"))
if len(videos) != 3 or any(path.stat().st_size == 0 for path in videos):
    raise SystemExit("final mock videos are missing")

audit = {
    "status": "PASS",
    "fresh_cosmos_evidence": report["evidence"],
    "inventory": dict(counts),
    "all_motion_targets_inside_policy_guards": True,
    "bin_center_errors_mm": {key: round(value, 3) for key, value in bin_errors.items()},
    "mock_videos": [str(path) for path in videos],
    "policy2_request_file_present": (POLICY / ".request.json").exists(),
}
if audit["policy2_request_file_present"]:
    raise SystemExit("policy 2 already has a request file; expected an unsubmitted policy")
(RUN / "policy2_final_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
