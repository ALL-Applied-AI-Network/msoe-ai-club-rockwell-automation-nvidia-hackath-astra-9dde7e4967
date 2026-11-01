#!/usr/bin/env python3
"""Build a bin-free mixed synthetic/real six-class image dataset."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path("/workspace/real_object_classifier_20261031")
SYNTHETIC = Path("/workspace/training_six_class_20261030/yolo_dataset")
REAL = ROOT / "real_crops"
OUTPUT = ROOT / "classifier_dataset"
CLASSES = (
    "m4_screw",
    "m8_screw",
    "round_pill",
    "long_pill",
    "aa_battery",
    "aaa_battery",
)
SYNTHETIC_PER_CLASS = 400
REAL_REPEATS_PER_CLASS = 400
SEED = 20261031


def crop_yolo_object(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    center_x, center_y, width, height = box
    width_px, height_px = image.size
    center_x *= width_px
    center_y *= height_px
    side = max(width * width_px, height * height_px) * 1.25
    side = max(24.0, side)
    bounds = (
        int(round(center_x - side / 2)),
        int(round(center_y - side / 2)),
        int(round(center_x + side / 2)),
        int(round(center_y + side / 2)),
    )
    return ImageOps.pad(image.crop(bounds), (224, 224), color=(235, 235, 235))


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (OUTPUT / split / class_name).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    candidates = defaultdict(list)
    image_root = SYNTHETIC / "images/train"
    label_root = SYNTHETIC / "labels/train"
    for label_path in sorted(label_root.glob("*.txt")):
        image_path = image_root / f"{label_path.stem}.png"
        if not image_path.is_file():
            continue
        for line_index, line in enumerate(label_path.read_text().splitlines()):
            values = line.split()
            if len(values) != 5:
                continue
            class_id = int(values[0])
            candidates[class_id].append(
                (image_path, line_index, tuple(float(value) for value in values[1:]))
            )

    counts = Counter()
    manifest = []
    for class_id, class_name in enumerate(CLASSES):
        choices = list(candidates[class_id])
        rng.shuffle(choices)
        for index, (image_path, line_index, box) in enumerate(choices[:SYNTHETIC_PER_CLASS]):
            with Image.open(image_path) as source:
                crop = crop_yolo_object(source.convert("RGB"), box)
            destination = OUTPUT / "train" / class_name / f"synthetic_{index:04d}.jpg"
            crop.save(destination, quality=95, subsampling=0)
            counts[("train", class_name, "synthetic")] += 1
            manifest.append(
                {
                    "split": "train",
                    "class": class_name,
                    "domain": "synthetic",
                    "source": str(image_path),
                    "dataset_file": str(destination.relative_to(OUTPUT)),
                    "bins_included": False,
                }
            )

    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            sources = sorted((REAL / split / class_name).glob("*.jpg"))
            if not sources:
                raise RuntimeError(f"No real {split} crops for {class_name}")
            copies = REAL_REPEATS_PER_CLASS if split == "train" else len(sources)
            for index in range(copies):
                source = sources[index % len(sources)]
                destination = OUTPUT / split / class_name / f"real_{index:04d}_{source.name}"
                link_or_copy(source, destination)
                counts[(split, class_name, "real")] += 1
                manifest.append(
                    {
                        "split": split,
                        "class": class_name,
                        "domain": "real",
                        "source": str(source),
                        "dataset_file": str(destination.relative_to(OUTPUT)),
                        "bins_included": False,
                    }
                )

    report = {
        "status": "complete",
        "classes": list(CLASSES),
        "synthetic_train_per_class": SYNTHETIC_PER_CLASS,
        "real_train_samples_per_class_after_sampling": REAL_REPEATS_PER_CLASS,
        "real_validation_and_test_are_not_duplicated": True,
        "bins_included": False,
        "counts": {
            split: {
                class_name: {
                    domain: counts[(split, class_name, domain)]
                    for domain in ("synthetic", "real")
                    if counts[(split, class_name, domain)]
                }
                for class_name in CLASSES
            }
            for split in ("train", "val", "test")
        },
    }
    (ROOT / "classifier_dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "classifier_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
