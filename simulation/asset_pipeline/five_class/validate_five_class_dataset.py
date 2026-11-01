"""Audit a generated five-class Isaac Replicator dataset."""

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
root = Path(args.dataset).resolve()

expected_classes = {"m4_screw", "m8_screw", "cube", "long_pill", "round_pill"}
rows = [
    json.loads(line)
    for line in (root / "scene_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
width, height = manifest["resolution"]
errors = []
class_instances = collections.Counter()
class_colors = collections.defaultdict(collections.Counter)
min_box = [width, height]
max_box = [0, 0]

if len(rows) != manifest["scenes"]:
    errors.append(f"metadata rows {len(rows)} != manifest scenes {manifest['scenes']}")
if set(manifest["semantic_classes"]) != expected_classes:
    errors.append(f"manifest classes are {manifest['semantic_classes']}")

patterns = {
    "rgb": "rgb_*.png",
    "bbox": "bounding_box_2d_tight_*.npy",
    "bbox_labels": "bounding_box_2d_tight_labels_*.json",
    "bbox_prim_paths": "bounding_box_2d_tight_prim_paths_*.json",
    "semantic_segmentation": "semantic_segmentation_*.png",
    "semantic_labels": "semantic_segmentation_labels_*.json",
}
file_counts = {name: len(list(root.glob(pattern))) for name, pattern in patterns.items()}
for name, count in file_counts.items():
    if count != len(rows):
        errors.append(f"{name} files {count} != metadata rows {len(rows)}")

for index, row in enumerate(rows):
    if row.get("frame_index") != index:
        errors.append(f"row {index} has frame_index {row.get('frame_index')}")
        continue
    suffix = f"{index:06d}"
    expected_count = sum(row["object_counts"].values())
    boxes = np.load(root / f"bounding_box_2d_tight_{suffix}.npy")
    if len(boxes) != expected_count:
        errors.append(f"frame {index}: {len(boxes)} boxes != {expected_count} objects")
    if len(row["objects"]) != expected_count:
        errors.append(f"frame {index}: metadata objects mismatch")

    label_map = json.loads(
        (root / f"bounding_box_2d_tight_labels_{suffix}.json").read_text(encoding="utf-8")
    )
    labels = {entry["class"] for entry in label_map.values()}
    if labels != expected_classes:
        errors.append(f"frame {index}: labels {sorted(labels)}")

    for box in boxes:
        x0, y0, x1, y1 = (int(box[name]) for name in ("x_min", "y_min", "x_max", "y_max"))
        min_box[0] = min(min_box[0], x1 - x0)
        min_box[1] = min(min_box[1], y1 - y0)
        max_box[0] = max(max_box[0], x1 - x0)
        max_box[1] = max(max_box[1], y1 - y0)
        if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
            errors.append(f"frame {index}: invalid box {(x0, y0, x1, y1)}")

    for obj in row["objects"]:
        class_name = obj["semantic_class"]
        class_instances[class_name] += 1
        class_colors[class_name][obj["color"]] += 1

# Decode representative first/middle/last images to catch corrupt PNG output.
sample_images = {}
for index in sorted({0, len(rows) // 2, len(rows) - 1}):
    path = root / f"rgb_{index:06d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    sample_images[str(index)] = None if image is None else [image.shape[1], image.shape[0]]
    if image is None or image.shape[:2] != (height, width):
        errors.append(f"frame {index}: RGB decode/size failure")

required_colors = {
    "m4_screw": {"black_steel_a", "black_steel_b", "black_steel_c"},
    "m8_screw": {"black_steel_a", "black_steel_b", "black_steel_c"},
    "cube": {"red", "blue", "yellow"},
    "long_pill": {"red", "yellow"},
    "round_pill": {"red", "yellow"},
}
for class_name, colors in required_colors.items():
    actual = set(class_colors[class_name])
    if actual != colors:
        errors.append(f"{class_name}: colors {sorted(actual)} != {sorted(colors)}")

report = {
    "status": "PASS" if not errors else "FAIL",
    "scene_count": len(rows),
    "resolution": [width, height],
    "file_counts": file_counts,
    "class_instances": dict(sorted(class_instances.items())),
    "class_color_instances": {
        name: dict(sorted(counts.items())) for name, counts in sorted(class_colors.items())
    },
    "box_size_pixels_min": min_box,
    "box_size_pixels_max": max_box,
    "sample_rgb_dimensions": sample_images,
    "errors": errors[:100],
}
(root / "dataset_validation.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
raise SystemExit(0 if not errors else 1)
