#!/usr/bin/env python3
"""Train, validate, and test the Astra six-class detector."""

import argparse
import json
import os
import time
from pathlib import Path


def json_value(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(output / "ultralytics_settings"))
    from ultralytics import YOLO

    progress_path = output / "progress.json"
    started = time.time()

    def write_progress(phase, current, total, **extra):
        payload = {
            "phase": phase,
            "current": int(current),
            "total": int(total),
            "percent": round(100.0 * current / total, 2) if total else 0.0,
            "elapsed_seconds": round(time.time() - started, 1),
            **extra,
        }
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    write_progress("setup", 0, args.epochs)
    model = YOLO(args.model)

    def on_epoch_end(trainer):
        metrics = {
            str(key): json_value(value)
            for key, value in getattr(trainer, "metrics", {}).items()
        }
        write_progress(
            "training",
            trainer.epoch + 1,
            args.epochs,
            metrics=metrics,
            estimated_total_seconds=(time.time() - started) * args.epochs / (trainer.epoch + 1),
        )

    model.add_callback("on_train_epoch_end", on_epoch_end)
    write_progress("training", 0, args.epochs)
    training_result = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(output),
        name="train",
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        patience=12,
        seed=20261030,
        deterministic=True,
        # The GB10/torch 2.13 development stack failed Ultralytics' AMP safety
        # comparison during the smoke test.  FP32 is slower but avoids NaNs and
        # zero-mAP checkpoints.
        amp=False,
        cos_lr=True,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.02,
        scale=0.10,
        fliplr=0.0,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.15,
        plots=True,
        save=True,
        save_period=5,
        val=True,
        verbose=True,
    )

    best_path = output / "train" / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"missing best checkpoint: {best_path}")
    best_model = YOLO(str(best_path))
    write_progress("validation", 0, 1, checkpoint=str(best_path))
    validation_result = best_model.val(
        data=str(Path(args.data).resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(output),
        name="final_validation",
        plots=True,
        verbose=True,
    )
    write_progress("validation", 1, 1, checkpoint=str(best_path))

    write_progress("testing", 0, 1, checkpoint=str(best_path))
    test_result = best_model.val(
        data=str(Path(args.data).resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(output),
        name="final_test",
        plots=True,
        verbose=True,
    )
    write_progress("testing", 1, 1, checkpoint=str(best_path))

    report = {
        "status": "COMPLETE",
        "model": args.model,
        "best_checkpoint": str(best_path),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "elapsed_seconds": round(time.time() - started, 1),
        "training_results": {
            str(key): json_value(value)
            for key, value in getattr(training_result, "results_dict", {}).items()
        },
        "validation_results": {
            str(key): json_value(value)
            for key, value in validation_result.results_dict.items()
        },
        "test_results": {
            str(key): json_value(value)
            for key, value in test_result.results_dict.items()
        },
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_progress("complete", 1, 1, report=str(output / "training_report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
