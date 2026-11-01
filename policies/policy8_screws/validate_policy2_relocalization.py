"""Offline replay/benchmark for Policy 2 per-pick object and bin refresh."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cosmos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.policy_root))
    from sort4_policy.controller import GRASP_Z, Sort4Controller
    from sort4_policy.instruction import default_task
    from sort4_policy.perception import Sort4CosmosClient

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source = json.loads(args.reference.read_text(encoding="utf-8"))
    client = Sort4CosmosClient()
    raw_cosmos = json.loads(args.cosmos.read_text(encoding="utf-8"))
    mapped = client._map_crop_scene(
        raw_cosmos, tuple(source["object_crop_pixels"]), rgb.shape
    )
    initial_start = time.perf_counter()
    mapped["objects"] = client._refine_objects(rgb, mapped.get("objects", []))
    mapped["bins"] = client._detect_bins(rgb)
    targets = client.to_targets(
        mapped,
        default_task(
            "Sort all M4 screws into the red bin, all M8 screws into the yellow "
            "bin, all red cubes into the green bin, and all blue cubes into the blue bin."
        ),
    )
    initial_local_ms = (time.perf_counter() - initial_start) * 1000.0
    start = time.perf_counter()
    refreshed = client.relocalize_targets(rgb, targets)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    for target in refreshed:
        Sort4Controller._validate_target(target)

    report = {
        "mode": "offline only; no cell client imported or contacted",
        "image": args.image.name,
        "targets": len(refreshed),
        "inventory": dict(Counter(target.sort_class for target in refreshed)),
        "initial_local_verification_ms": initial_local_ms,
        "max_initial_refinement_mm": max(
            target.initial_refinement_mm for target in targets
        ),
        "max_initial_yaw_refinement_deg": max(
            target.initial_yaw_refinement_deg for target in targets
        ),
        "elapsed_ms": elapsed_ms,
        "per_target_ms": elapsed_ms / len(refreshed),
        "max_object_shift_mm": max(
            target.relocalization_mm for target in refreshed
        ),
        "max_yaw_jitter_deg": max(
            target.relocalization_yaw_deg for target in refreshed
        ),
        "max_bin_shift_mm": max(
            target.bin_relocalization_mm for target in refreshed
        ),
        "grasp_z_mm": GRASP_Z,
        "commands": [
            {
                "class": target.sort_class,
                "xy_mm": np.round(target.object_xy_mm, 3).tolist(),
                "yaw_deg": round(target.object_yaw_deg, 3),
                "bin_color": target.bin_color,
                "bin_xy_mm": np.round(target.bin_xy_mm, 3).tolist(),
                "object_shift_mm": round(target.relocalization_mm, 4),
                "yaw_shift_deg": round(target.relocalization_yaw_deg, 4),
                "bin_shift_mm": round(target.bin_relocalization_mm, 4),
            }
            for target in refreshed
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
