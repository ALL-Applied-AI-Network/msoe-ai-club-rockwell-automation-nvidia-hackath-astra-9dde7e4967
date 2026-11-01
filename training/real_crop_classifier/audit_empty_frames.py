#!/usr/bin/env python3
"""Audit completed-run frames for false full-frame object proposals."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from full_frame_detector import (
    ALLOWED,
    CAMERA_FILE,
    MODEL,
    OUTPUT,
    PAIRWISE_CONFIDENCE_THRESHOLD,
    ProposalConfig,
    bin_boxes,
    camera_occlusion,
    project_table,
    proposals,
    square_crop,
)


ROOT = Path("/workspace/real_object_classifier_20261031")


def completed_sessions() -> list[dict]:
    sessions = []
    for policy in (8, 9, 10):
        runtime = Path(f"/workspace/physical_ai_submission_policy{policy}/runtime_logs")
        for log_path in sorted(runtime.glob("policy*_20261030*/r20261030-*.jsonl")):
            inventory = None
            completed = {}
            ignored = []
            explicit_complete = False
            for line in log_path.read_text(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "batch_plan":
                    inventory = event.get("inventory")
                    ignored = event.get("ignored_objects") or []
                if isinstance(event.get("completed"), dict):
                    completed = event["completed"]
                explicit_complete |= event.get("event") == "complete"
            all_sorted = bool(inventory) and all(
                int(completed.get(key, 0)) >= int(value) for key, value in inventory.items()
            )
            if not (explicit_complete or all_sorted):
                continue
            frame_dir = log_path.with_name(log_path.stem + "_frames")
            frame_paths = sorted(frame_dir.glob("frame_*.png"))
            if not frame_paths:
                continue
            sessions.append(
                {
                    "policy": policy,
                    "session": log_path.parent.name,
                    "trial": log_path.stem,
                    "frame": frame_paths[-1],
                    "inventory": inventory,
                    "completed": completed,
                    "ignored_objects": ignored,
                    "explicit_complete": explicit_complete,
                }
            )
    return sessions


def main() -> None:
    full_report = json.loads((OUTPUT / "full_frame_report.json").read_text())
    config = ProposalConfig(**full_report["best_config"])
    camera = json.loads(CAMERA_FILE.read_text())["table"]
    workspace_polygon = np.round(
        project_table(
            [(x, y, -0.0051) for x, y in ((0.27, -0.28), (0.50, -0.28), (0.50, 0.28), (0.27, 0.28))],
            camera,
        )
    ).astype(np.int32)
    model = YOLO(str(MODEL))
    names = {int(key): value for key, value in model.names.items()}
    name_to_id = {value: key for key, value in names.items()}
    audit_dir = OUTPUT / "empty_audit_annotated"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = Counter()

    for session in completed_sessions():
        image = cv2.imread(str(session["frame"]))
        blocked, occluder_box, occluder_area = camera_occlusion(
            image, workspace_polygon
        )
        if blocked:
            detected = []
            height, width = image.shape[:2]
            workspace = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(workspace, workspace_polygon, 255)
            bins = bin_boxes(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), workspace)
        else:
            detected, bins = proposals(image, config, workspace_polygon)
        crops = [square_crop(image, item["box"]) for item in detected]
        predictions = model.predict(crops, imgsz=224, batch=max(1, len(crops)), device=0, verbose=False) if crops else []
        allowed_ids = [name_to_id[name] for name in ALLOWED[session["policy"]]]
        for item, prediction in zip(detected, predictions):
            probabilities = prediction.probs.data.cpu().numpy()
            class_id = max(allowed_ids, key=lambda index: probabilities[index])
            item["class"] = names[class_id]
            item["confidence"] = float(probabilities[class_id])
            allowed_mass = float(sum(probabilities[index] for index in allowed_ids))
            item["pairwise_confidence"] = (
                item["confidence"] / allowed_mass if allowed_mass > 0 else 0.0
            )
        counts.update(frames=1, camera_blocked=int(blocked), proposals=len(detected))
        for threshold in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90):
            counts[f"accepted_{threshold:.2f}"] += sum(
                item["pairwise_confidence"] >= threshold for item in detected
            )

        annotated = image.copy()
        if blocked and occluder_box is not None:
            x0, y0, x1, y1 = occluder_box
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 4)
            cv2.putText(
                annotated,
                f"CAMERA BLOCKED ({occluder_area:.0f}px)",
                (x0, max(28, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        for x0, y0, x1, y1 in bins:
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
        for item in detected:
            x0, y0, x1, y1 = item["box"]
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"{item['class']} pair={item['pairwise_confidence']:.2f}",
                (x0, max(18, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        destination = audit_dir / f"{session['session']}__{session['trial']}.jpg"
        cv2.imwrite(str(destination), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(
            {
                **{key: value for key, value in session.items() if key != "frame"},
                "frame": str(session["frame"]),
                "camera_blocked": blocked,
                "occluder_box": occluder_box,
                "occluder_area": occluder_area,
                "proposals": detected,
                "annotated": str(destination.relative_to(ROOT)),
            }
        )

    report = {
        "status": "complete",
        "assumption": "last frame after policy completed its full recorded inventory has no remaining target objects",
        "frames": counts["frames"],
        "camera_blocked_frames": counts["camera_blocked"],
        "pairwise_confidence_threshold": PAIRWISE_CONFIDENCE_THRESHOLD,
        "raw_proposals": counts["proposals"],
        "raw_proposals_per_frame": counts["proposals"] / max(1, counts["frames"]),
        "accepted_false_proposals": {
            threshold: counts[f"accepted_{threshold:.2f}"]
            for threshold in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
        },
        "sessions": rows,
    }
    path = OUTPUT / "empty_frame_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
