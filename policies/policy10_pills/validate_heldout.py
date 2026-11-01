#!/usr/bin/env python3
"""Replay the complete policy perception gate on held-out synthetic scenes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from family_policy.config import MODEL_CLASSES
from family_policy.perception import TrainedDetectorClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    coco = json.loads((args.dataset / "annotations_test.json").read_text())
    categories = {int(row["id"]): row["name"] for row in coco["categories"]}
    annotations = defaultdict(list)
    for row in coco["annotations"]:
        annotations[int(row["image_id"])].append(row)
    client = TrainedDetectorClient()
    rows = []
    for image_row in sorted(coco["images"], key=lambda item: int(item["id"]))[: args.limit]:
        path = args.dataset / image_row["file_name"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"could not read {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _, (left, top, right, bottom) = client._object_crop(image)
        expected = Counter()
        for annotation in annotations[int(image_row["id"])]:
            name = categories[int(annotation["category_id"])]
            x, y, box_width, box_height = annotation["bbox"]
            center_x, center_y = x + box_width / 2, y + box_height / 2
            if name in MODEL_CLASSES and left <= center_x <= right and top <= center_y <= bottom:
                expected[name] += 1
        actual = Counter(
            item["sort_class"] for item in client.detect_objects(image, allow_empty=True)
        )
        rows.append(
            {
                "image": image_row["file_name"],
                "expected": dict(expected),
                "actual": dict(actual),
                "exact": expected == actual,
            }
        )
    report = {
        "checkpoint_gate": "trained proposals plus metric verification",
        "classes": list(MODEL_CLASSES),
        "scenes": len(rows),
        "exact_scenes": sum(row["exact"] for row in rows),
        "rows": rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
