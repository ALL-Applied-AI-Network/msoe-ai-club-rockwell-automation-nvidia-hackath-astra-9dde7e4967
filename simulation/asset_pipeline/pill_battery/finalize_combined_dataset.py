#!/usr/bin/env python3
"""Validate the six-class Replicator dataset and export COCO annotations."""

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np


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
SPLITS = ("train", "validation", "test")


def png_dimensions(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG header")
    return struct.unpack(">II", header[16:24])


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    errors = []
    warnings = []

    manifest_path = root / "dataset_manifest.json"
    metadata_path = root / "scene_metadata.jsonl"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise SystemExit("dataset_manifest.json or scene_metadata.jsonl is missing")
    manifest = load_json(manifest_path)
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_scenes = int(manifest["scenes"])
    width, height = map(int, manifest["resolution"])
    if tuple(manifest.get("semantic_classes", ())) != CLASSES:
        errors.append("manifest semantic class order is not the approved six-class order")
    if len(rows) != expected_scenes:
        errors.append(f"metadata has {len(rows)} rows; expected {expected_scenes}")
    if any(root.rglob("*.mp4")):
        errors.append("video files are present, but this dataset must not generate videos")

    categories = [
        {"id": index + 1, "name": name, "supercategory": next(k for k, v in FAMILIES.items() if name in v)}
        for index, name in enumerate(CLASSES)
    ]
    category_ids = {row["name"]: row["id"] for row in categories}
    coco = {
        split: {
            "info": {"description": "Astra CSI six-class Isaac Sim dataset", "version": "1.0"},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": [],
        }
        for split in SPLITS
    }
    instance_counts = collections.Counter()
    scene_presence = collections.Counter()
    absent_scenes = collections.Counter()
    color_counts = {name: collections.Counter() for name in CLASSES}
    lighting_counts = collections.Counter()
    split_counts = collections.Counter()
    total_boxes = 0
    next_annotation_id = {split: 1 for split in SPLITS}

    for ordinal, row in enumerate(rows):
        frame = int(row["frame_index"])
        if frame != ordinal:
            errors.append(f"metadata row {ordinal} has frame_index {frame}")
        split = row.get("split")
        if split not in SPLITS:
            errors.append(f"frame {frame}: invalid split {split!r}")
            continue
        split_counts[split] += 1
        counts = row.get("object_counts", {})
        if set(counts) != set(CLASSES):
            errors.append(f"frame {frame}: object_counts does not contain exactly six classes")
        for family, members in FAMILIES.items():
            if sum(int(counts.get(name, 0)) for name in members) < 1:
                errors.append(f"frame {frame}: no {family} object")
        expected_objects = sum(int(counts.get(name, 0)) for name in CLASSES)
        objects = row.get("objects", [])
        if len(objects) != expected_objects:
            errors.append(f"frame {frame}: metadata object list/count mismatch")
        expected_paths = {f"/World/DatasetObjects/{obj['name']}" for obj in objects}
        for name in CLASSES:
            value = int(counts.get(name, 0))
            instance_counts[name] += value
            if value:
                scene_presence[name] += 1
            else:
                absent_scenes[name] += 1
        for obj in objects:
            name = obj.get("semantic_class")
            if name not in CLASSES:
                errors.append(f"frame {frame}: invalid object class {name!r}")
            else:
                color_counts[name][obj.get("color", "<missing>")] += 1
        lighting_counts[row.get("lighting", {}).get("profile", "<missing>")] += 1

        stem = f"{frame:06d}"
        rgb_path = root / row.get("rgb_file", f"rgb_{stem}.png")
        seg_path = root / row.get("semantic_segmentation_file", f"semantic_segmentation_{stem}.png")
        bbox_path = root / row.get("bbox_file", f"bounding_box_2d_tight_{stem}.npy")
        label_path = root / f"bounding_box_2d_tight_labels_{stem}.json"
        prim_path = root / f"bounding_box_2d_tight_prim_paths_{stem}.json"
        seg_label_path = root / f"semantic_segmentation_labels_{stem}.json"
        for path in (rgb_path, seg_path, bbox_path, label_path, prim_path, seg_label_path):
            if not path.is_file():
                errors.append(f"frame {frame}: missing {path.name}")
        if any(not path.is_file() for path in (rgb_path, seg_path, bbox_path, label_path, prim_path, seg_label_path)):
            continue
        try:
            if png_dimensions(rgb_path) != (width, height):
                errors.append(f"frame {frame}: RGB resolution mismatch")
            if png_dimensions(seg_path) != (width, height):
                errors.append(f"frame {frame}: segmentation resolution mismatch")
        except Exception as exc:
            errors.append(f"frame {frame}: PNG validation failed: {exc}")

        boxes = np.load(bbox_path, allow_pickle=False)
        labels = load_json(label_path)
        prim_paths = load_json(prim_path)
        if len(boxes) != expected_objects or len(prim_paths) != expected_objects:
            errors.append(
                f"frame {frame}: expected {expected_objects} boxes/paths, got {len(boxes)}/{len(prim_paths)}"
            )
        if set(prim_paths) != expected_paths:
            errors.append(f"frame {frame}: annotated prim paths do not match metadata objects")
        label_classes = {item.get("class") for item in labels.values()}
        if not label_classes.issubset(set(CLASSES)):
            errors.append(f"frame {frame}: bbox label map contains an unknown class")

        coco[split]["images"].append(
            {"id": frame + 1, "file_name": rgb_path.name, "width": width, "height": height}
        )
        for box in boxes:
            semantic_id = str(int(box["semanticId"]))
            class_name = labels.get(semantic_id, {}).get("class")
            if class_name not in CLASSES:
                errors.append(f"frame {frame}: semantic ID {semantic_id} has no approved class")
                continue
            x0 = max(0, min(width, int(box["x_min"])))
            y0 = max(0, min(height, int(box["y_min"])))
            x1 = max(0, min(width, int(box["x_max"])))
            y1 = max(0, min(height, int(box["y_max"])))
            box_width = x1 - x0
            box_height = y1 - y0
            if box_width <= 0 or box_height <= 0:
                errors.append(f"frame {frame}: non-positive box for {class_name}")
                continue
            annotation = {
                "id": next_annotation_id[split],
                "image_id": frame + 1,
                "category_id": category_ids[class_name],
                "bbox": [x0, y0, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
            if "occlusionRatio" in boxes.dtype.names:
                annotation["attributes"] = {"occlusion_ratio": float(box["occlusionRatio"])}
            coco[split]["annotations"].append(annotation)
            next_annotation_id[split] += 1
            total_boxes += 1

    for split in SPLITS:
        listed = [x for x in (root / f"{split}.txt").read_text(encoding="utf-8").splitlines() if x]
        generated = [item["file_name"] for item in coco[split]["images"]]
        if listed != generated:
            errors.append(f"{split}.txt does not match metadata ordering")
        with (root / f"annotations_{split}.json").open("w", encoding="utf-8") as stream:
            json.dump(coco[split], stream, separators=(",", ":"))

    for name in CLASSES:
        if absent_scenes[name] == 0:
            warnings.append(f"{name} is present in every scene; no class-absence negatives")
        if len(color_counts[name]) < 3:
            warnings.append(f"{name} has fewer than three appearance colors")
    expected_split_counts = manifest.get("splits", {})
    if dict(split_counts) != expected_split_counts:
        errors.append(f"split counts {dict(split_counts)} do not match manifest {expected_split_counts}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(root),
        "scenes": len(rows),
        "resolution": [width, height],
        "total_boxes": total_boxes,
        "semantic_classes": list(CLASSES),
        "split_scene_counts": dict(split_counts),
        "split_annotation_counts": {split: len(coco[split]["annotations"]) for split in SPLITS},
        "class_instance_counts": dict(instance_counts),
        "class_scene_presence": dict(scene_presence),
        "class_absent_scene_counts": dict(absent_scenes),
        "appearance_color_counts": {name: dict(values) for name, values in color_counts.items()},
        "lighting_profile_counts": dict(lighting_counts),
        "video_count": len(list(root.rglob("*.mp4"))),
        "coco_exports": [f"annotations_{split}.json" for split in SPLITS],
        "errors": errors,
        "warnings": warnings,
    }
    with (root / "dataset_validation.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
