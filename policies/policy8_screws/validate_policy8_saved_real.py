#!/usr/bin/env python3
"""Run Policy 8's local screw inventory on saved real CSI frames (no motion)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from sort4_policy.instruction import default_task
from sort4_policy.perception import Sort4CosmosClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    client = Sort4CosmosClient("http://127.0.0.1:9")
    task = default_task(
        "Sort all M4 screws into the red bin and all M8 screws into the green bin."
    )
    report = []
    for path in args.images:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Could not read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        proposals = client._pixel_screw_proposals(rgb)
        objects = client._refine_objects(rgb, [], allow_empty=True)
        counts = Counter(item["sort_class"] for item in objects)
        report.append(
            {
                "image": str(path),
                "full_field_proposals": len(proposals),
                "verified_counts": dict(counts),
                "verified_screws": [
                    {
                        "class": item["sort_class"],
                        "center": item["center"],
                        "head_width_mm": round(float(item["head_width_mm"]), 2),
                        "shank_width_mm": round(float(item["shank_width_mm"]), 2),
                        "length_mm": round(float(item["screw_length_mm"]), 2),
                    }
                    for item in objects
                ],
                "task_mapping": task.class_to_bin,
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
