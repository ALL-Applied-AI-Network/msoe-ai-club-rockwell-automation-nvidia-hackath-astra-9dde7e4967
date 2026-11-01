#!/usr/bin/env python3
"""Convert the validated COCO exports into a zero-copy YOLO dataset."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import yaml


CLASSES = (
    "m4_screw",
    "m8_screw",
    "round_pill",
    "long_pill",
    "aa_battery",
    "aaa_battery",
)
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((source / "dataset_validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise SystemExit("source dataset has not passed validation")
    if tuple(manifest.get("semantic_classes", ())) != CLASSES:
        raise SystemExit("source class order does not match the approved six classes")

    summary = {"source": str(source), "output": str(out), "classes": list(CLASSES), "splits": {}}
    for source_split, yolo_split in SPLIT_MAP.items():
        coco_path = source / f"annotations_{source_split}.json"
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {int(row["id"]): row["name"] for row in coco["categories"]}
        class_ids = {category_id: CLASSES.index(name) for category_id, name in categories.items()}
        annotations = {}
        for row in coco["annotations"]:
            annotations.setdefault(int(row["image_id"]), []).append(row)

        image_dir = out / "images" / yolo_split
        label_dir = out / "labels" / yolo_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        class_counts = Counter()
        empty_images = 0
        for image in coco["images"]:
            image_id = int(image["id"])
            width = float(image["width"])
            height = float(image["height"])
            source_image = source / image["file_name"]
            if not source_image.is_file():
                raise SystemExit(f"missing image: {source_image}")
            link = image_dir / source_image.name
            relative_target = os.path.relpath(source_image, link.parent)
            link.symlink_to(relative_target)

            lines = []
            for annotation in annotations.get(image_id, []):
                x, y, box_width, box_height = map(float, annotation["bbox"])
                category_id = int(annotation["category_id"])
                class_id = class_ids[category_id]
                center_x = (x + box_width / 2.0) / width
                center_y = (y + box_height / 2.0) / height
                normalized_width = box_width / width
                normalized_height = box_height / height
                values = (center_x, center_y, normalized_width, normalized_height)
                if not all(0.0 <= value <= 1.0 for value in values):
                    raise SystemExit(f"out-of-range box in {source_image.name}: {values}")
                lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in values))
                class_counts[CLASSES[class_id]] += 1
            if not lines:
                empty_images += 1
            (label_dir / f"{source_image.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        summary["splits"][yolo_split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "empty_images": empty_images,
            "class_counts": dict(class_counts),
        }

    dataset_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASSES)},
    }
    (out / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    (out / "conversion_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
