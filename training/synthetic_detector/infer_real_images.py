#!/usr/bin/env python3
"""Run the trained six-class detector on unlabeled real CSI images."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


THRESHOLDS = (0.05, 0.10, 0.25, 0.50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    started = time.time()
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=min(THRESHOLDS),
        iou=0.50,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(args.output),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )

    class_names = {int(key): value for key, value in model.names.items()}
    totals = {str(threshold): Counter() for threshold in THRESHOLDS}
    image_hits = {str(threshold): Counter() for threshold in THRESHOLDS}
    per_image = []
    confidence_values = defaultdict(list)

    for result in results:
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for box, class_id_raw, confidence_raw in zip(boxes, classes, confidences):
                class_id = int(class_id_raw)
                confidence = float(confidence_raw)
                class_name = class_names[class_id]
                confidence_values[class_name].append(confidence)
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "xyxy": [float(value) for value in box],
                    }
                )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        per_image.append({"image": Path(result.path).name, "detections": detections})

        for threshold in THRESHOLDS:
            threshold_key = str(threshold)
            classes_present = set()
            for detection in detections:
                if detection["confidence"] >= threshold:
                    class_name = detection["class_name"]
                    totals[threshold_key][class_name] += 1
                    classes_present.add(class_name)
            for class_name in classes_present:
                image_hits[threshold_key][class_name] += 1

    rows = []
    for entry in per_image:
        for detection in entry["detections"]:
            rows.append({"image": entry["image"], **detection})
    with (args.output / "detections.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("image", "class_id", "class_name", "confidence", "xyxy"),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "COMPLETE",
        "model": str(args.model),
        "source": str(args.source),
        "images": len(per_image),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "elapsed_seconds": round(time.time() - started, 3),
        "thresholds": {
            str(threshold): {
                "detections_by_class": dict(totals[str(threshold)]),
                "images_with_class": dict(image_hits[str(threshold)]),
            }
            for threshold in THRESHOLDS
        },
        "confidence_by_class": {
            class_name: {
                "count": len(values),
                "minimum": min(values),
                "mean": sum(values) / len(values),
                "maximum": max(values),
            }
            for class_name, values in sorted(confidence_values.items())
            if values
        },
        "per_image": per_image,
    }
    (args.output / "inference_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
