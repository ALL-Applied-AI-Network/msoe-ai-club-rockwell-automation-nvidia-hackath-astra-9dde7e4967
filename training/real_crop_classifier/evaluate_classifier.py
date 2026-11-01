#!/usr/bin/env python3
"""Produce transparent per-image and per-class real-test results."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


ROOT = Path("/workspace/real_object_classifier_20261031")
MODEL = ROOT / "training/run/weights/best.pt"
TEST = ROOT / "classifier_dataset/test"


def main() -> None:
    model = YOLO(str(MODEL))
    files = sorted(TEST.glob("*/*"))
    results = model.predict([str(path) for path in files], imgsz=224, batch=64, device=0, verbose=False)
    rows = []
    confusion = defaultdict(Counter)
    for path, result in zip(files, results):
        true_class = path.parent.name
        predicted_id = int(result.probs.top1)
        predicted_class = result.names[predicted_id]
        confidence = float(result.probs.top1conf)
        correct = true_class == predicted_class
        frame = "frame_00000" if "frame_00000" in path.name else "frame_00001"
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 8),
                "correct": correct,
                "frame": frame,
            }
        )
        confusion[true_class][predicted_class] += 1

    def summary(selected: list[dict]) -> dict:
        by_class = {}
        for class_name in sorted(model.names.values()):
            class_rows = [row for row in selected if row["true_class"] == class_name]
            by_class[class_name] = {
                "correct": sum(row["correct"] for row in class_rows),
                "total": len(class_rows),
                "accuracy": (
                    sum(row["correct"] for row in class_rows) / len(class_rows)
                    if class_rows
                    else None
                ),
            }
        return {
            "correct": sum(row["correct"] for row in selected),
            "total": len(selected),
            "accuracy": sum(row["correct"] for row in selected) / len(selected),
            "by_class": by_class,
        }

    report = {
        "status": "complete",
        "model": str(MODEL),
        "bins_in_training_or_test": False,
        "all_test_crops": summary(rows),
        "frame_00000_only": summary([row for row in rows if row["frame"] == "frame_00000"]),
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
        "misclassifications": [row for row in rows if not row["correct"]],
        "predictions": rows,
    }
    (ROOT / "classifier_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "classifier_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
