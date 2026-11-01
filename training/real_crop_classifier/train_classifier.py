#!/usr/bin/env python3
"""Train and evaluate the bin-free six-class object classifier."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ultralytics import YOLO


ROOT = Path("/workspace/real_object_classifier_20261031")
DATASET = ROOT / "classifier_dataset"
OUTPUT = ROOT / "training"
DETECTOR_WEIGHTS = Path("/workspace/training_six_class_20261030/yolo11s.pt")
EPOCHS = 35


def serializable(values: dict) -> dict:
    return {
        str(key): value.item() if hasattr(value, "item") else value
        for key, value in values.items()
        if isinstance(value, (str, int, float, bool)) or hasattr(value, "item")
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    progress_path = ROOT / "training_progress.json"
    started = time.time()

    def progress(phase: str, current: int, total: int, **extra) -> None:
        payload = {
            "phase": phase,
            "current": current,
            "total": total,
            "percent": round(100.0 * current / total, 2) if total else 0.0,
            "elapsed_seconds": round(time.time() - started, 1),
            **extra,
        }
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(progress_path)

    # Reuse the already available COCO-pretrained YOLO11s backbone; this avoids
    # a network download while replacing the detection head with classification.
    model = YOLO("yolo11s-cls.yaml").load(str(DETECTOR_WEIGHTS))

    def epoch_end(trainer) -> None:
        progress(
            "training",
            trainer.epoch + 1,
            EPOCHS,
            metrics=serializable(getattr(trainer, "metrics", {})),
        )

    model.add_callback("on_train_epoch_end", epoch_end)
    progress("training", 0, EPOCHS)
    train_result = model.train(
        data=str(DATASET),
        epochs=EPOCHS,
        imgsz=224,
        batch=64,
        device=0,
        workers=8,
        project=str(OUTPUT),
        name="run",
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        weight_decay=0.0005,
        patience=10,
        seed=20261031,
        deterministic=True,
        amp=False,
        degrees=180.0,
        translate=0.04,
        scale=0.15,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.20,
        auto_augment="randaugment",
        erasing=0.15,
        plots=True,
        save=True,
        verbose=True,
    )

    best = OUTPUT / "run/weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Missing checkpoint {best}")
    trained = YOLO(str(best))

    progress("validation", 0, 1, checkpoint=str(best))
    validation = trained.val(
        data=str(DATASET),
        split="val",
        imgsz=224,
        batch=64,
        device=0,
        workers=8,
        project=str(OUTPUT),
        name="real_validation",
        plots=True,
        verbose=True,
    )
    progress("validation", 1, 1, checkpoint=str(best))

    progress("testing", 0, 1, checkpoint=str(best))
    testing = trained.val(
        data=str(DATASET),
        split="test",
        imgsz=224,
        batch=64,
        device=0,
        workers=8,
        project=str(OUTPUT),
        name="real_test",
        plots=True,
        verbose=True,
    )
    progress("testing", 1, 1, checkpoint=str(best))

    report = {
        "status": "complete",
        "model": "YOLO11s classification head with transferred detector backbone",
        "classes": list(trained.names.values()),
        "epochs_requested": EPOCHS,
        "checkpoint": str(best),
        "bins_in_training": False,
        "elapsed_seconds": round(time.time() - started, 1),
        "training_metrics": serializable(getattr(train_result, "results_dict", {})),
        "real_validation_metrics": serializable(validation.results_dict),
        "real_test_metrics": serializable(testing.results_dict),
    }
    report_path = ROOT / "classifier_training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    progress("complete", 1, 1, report=str(report_path), checkpoint=str(best))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
