#!/usr/bin/env python3
"""Evaluate bin-masked full-frame proposals followed by the six-class classifier."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path("/workspace/real_object_classifier_20261031")
MANIFEST = ROOT / "real_crop_manifest.csv"
MODEL = ROOT / "training/run/weights/best.pt"
CAMERA_FILE = Path(
    "/workspace/physical_ai_submission_policy10/calibration/cameras.json"
)
OUTPUT = ROOT / "full_frame_evaluation"
# The six-way softmax is deliberately reduced to the two classes that can occur
# in the current task.  This avoids rejecting a dim real object merely because
# an impossible class received probability mass.
PAIRWISE_CONFIDENCE_THRESHOLD = 0.75
ALLOWED = {
    8: ("m4_screw", "m8_screw"),
    9: ("aa_battery", "aaa_battery"),
    10: ("round_pill", "long_pill"),
}


@dataclass(frozen=True)
class ProposalConfig:
    dark_v: int
    saturation: int
    min_area: int
    max_area: int
    max_dimension: int


def project_table(points: list[tuple[float, float, float]], camera: dict) -> np.ndarray:
    transform = np.asarray(camera["T_base_cam"], dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    pts = np.asarray(points, dtype=float)
    camera_points = (pts - translation) @ rotation
    x = camera_points[:, 0] / camera_points[:, 2]
    y = camera_points[:, 1] / camera_points[:, 2]
    distortion = 1.0 + camera["k1"] * (x * x + y * y)
    width, height = camera["size"]
    return np.stack(
        [camera["fx"] * x * distortion + width / 2, camera["fx"] * y * distortion + height / 2],
        axis=1,
    )


def image_path(row: dict) -> Path:
    policy = int(row["session"].split("_", 1)[0].removeprefix("policy"))
    return Path(
        f"/workspace/physical_ai_submission_policy{policy}/runtime_logs/"
        f"{row['session']}/{row['trial']}_frames/{row['frame']}"
    )


def load_frames() -> dict[tuple[str, str], dict]:
    frames = {}
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["session"], row["frame"])
            entry = frames.setdefault(
                key,
                {
                    "session": row["session"],
                    "trial": row["trial"],
                    "frame": row["frame"],
                    "split": row["split"],
                    "path": image_path(row),
                    "policy": int(row["session"].split("_", 1)[0].removeprefix("policy")),
                    "targets": [],
                },
            )
            entry["targets"].append(
                {
                    "class": row["class"],
                    "center": (float(row["center_x"]), float(row["center_y"])),
                }
            )
    missing = [str(entry["path"]) for entry in frames.values() if not entry["path"].is_file()]
    if missing:
        raise RuntimeError(f"Missing source frames: {missing[:5]}")
    return frames


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    width = max(0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = width * height
    if not intersection:
        return 0.0
    return intersection / min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))


def bin_boxes(hsv: np.ndarray, workspace: np.ndarray) -> list[tuple[int, int, int, int]]:
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.where((saturation >= 75) & (value >= 55), 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, workspace)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    boxes = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if area >= 500 and width >= 28 and height >= 28:
            margin = 12
            boxes.append((x - margin, y - margin, x + width + margin, y + height + margin))
    return boxes


def camera_occlusion(
    image: np.ndarray, workspace_polygon: np.ndarray
) -> tuple[bool, tuple[int, int, int, int] | None, float]:
    """Fail closed when the wrist/gripper reaches into the loose-object ROI.

    The scan pose leaves a visible gap above the projected table workspace.
    During/after a pick, the robot and its cables form a much larger foreground
    component that bridges that gap.  Small loose objects are far below the
    component-area threshold, so they cannot trip this guard by themselves.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    foreground = np.where(
        (value <= 125) | ((saturation >= 55) & (value >= 40)), 255, 0
    ).astype(np.uint8)

    top_y = int(np.min(workspace_polygon[:, 1]))
    left_x = max(0, int(np.min(workspace_polygon[:, 0])) - 80)
    right_x = min(width, int(np.max(workspace_polygon[:, 0])) + 80)
    corridor = np.zeros((height, width), np.uint8)
    cv2.rectangle(
        corridor,
        (left_x, max(0, top_y - 230)),
        (right_x, min(height - 1, top_y + 105)),
        255,
        -1,
    )
    foreground = cv2.bitwise_and(foreground, corridor)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8)
    )

    best_box = None
    best_area = 0.0
    for contour in cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        area = float(cv2.contourArea(contour))
        x, y, box_width, box_height = cv2.boundingRect(contour)
        # It must be a substantial component that starts above the workspace
        # and reaches into (or to the lip of) the projected object region.
        bridges_boundary = y <= top_y - 70 and y + box_height >= top_y - 8
        if area >= 1800 and bridges_boundary and area > best_area:
            best_area = area
            best_box = (x, y, x + box_width, y + box_height)
    return best_box is not None, best_box, best_area


def proposals(
    image: np.ndarray, config: ProposalConfig, workspace_polygon: np.ndarray
) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
    height, width = image.shape[:2]
    workspace = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(workspace, workspace_polygon, 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins = bin_boxes(hsv, workspace)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark = value <= config.dark_v
    chromatic = (saturation >= config.saturation) & (value >= 45)
    mask = np.where(dark | chromatic, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, workspace)
    for x0, y0, x1, y1 in bins:
        cv2.rectangle(mask, (max(0, x0), max(0, y0)), (min(width - 1, x1), min(height - 1, y1)), 0, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    found = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        x, y, box_width, box_height = cv2.boundingRect(contour)
        long_side = max(box_width, box_height)
        short_side = min(box_width, box_height)
        if not (config.min_area <= area <= config.max_area):
            continue
        if long_side > config.max_dimension or short_side < 3:
            continue
        margin = max(4, int(round(0.12 * long_side)))
        box = (
            max(0, x - margin),
            max(0, y - margin),
            min(width, x + box_width + margin),
            min(height, y + box_height + margin),
        )
        found.append(
            {
                "box": box,
                "center": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                "area": area,
                "long_side": long_side,
                "short_side": short_side,
            }
        )

    found.sort(key=lambda item: item["area"], reverse=True)
    kept = []
    for item in found:
        if any(overlap(item["box"], prior["box"]) > 0.55 for prior in kept):
            continue
        kept.append(item)
    kept.sort(key=lambda item: (item["center"][1], item["center"][0]))
    return kept, bins


def match_targets(targets: list[dict], detected: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pairs = []
    for target_index, target in enumerate(targets):
        for proposal_index, proposal in enumerate(detected):
            distance = math.dist(target["center"], proposal["center"])
            x0, y0, x1, y1 = proposal["box"]
            inside = x0 - 10 <= target["center"][0] <= x1 + 10 and y0 - 10 <= target["center"][1] <= y1 + 10
            if inside or distance <= 32:
                pairs.append((distance, target_index, proposal_index))
    pairs.sort()
    matched_targets, matched_proposals, matches = set(), set(), []
    for _, target_index, proposal_index in pairs:
        if target_index in matched_targets or proposal_index in matched_proposals:
            continue
        matched_targets.add(target_index)
        matched_proposals.add(proposal_index)
        matches.append((target_index, proposal_index))
    return (
        matches,
        [index for index in range(len(targets)) if index not in matched_targets],
        [index for index in range(len(detected)) if index not in matched_proposals],
    )


def proposal_metrics(frames: list[dict], config: ProposalConfig, workspace_polygon: np.ndarray) -> dict:
    counts = Counter()
    for frame in frames:
        image = cv2.imread(str(frame["path"]))
        detected, _ = proposals(image, config, workspace_polygon)
        matches, misses, extras = match_targets(frame["targets"], detected)
        counts.update(tp=len(matches), fn=len(misses), fp=len(extras), images=1)
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_proposals_per_image": counts["fp"] / max(1, counts["images"]),
    }


def square_crop(image: np.ndarray, box: tuple[int, int, int, int], side: int = 96) -> np.ndarray:
    x0, y0, x1, y1 = box
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = side / 2.0
    left, top = int(round(center_x - half)), int(round(center_y - half))
    right, bottom = left + side, top + side
    result = np.full((side, side, 3), 235, dtype=np.uint8)
    source_x0, source_y0 = max(0, left), max(0, top)
    source_x1, source_y1 = min(image.shape[1], right), min(image.shape[0], bottom)
    result[source_y0 - top : source_y1 - top, source_x0 - left : source_x1 - left] = image[
        source_y0:source_y1, source_x0:source_x1
    ]
    return result


def evaluate(
    frames: list[dict], config: ProposalConfig, workspace_polygon: np.ndarray, model: YOLO, split: str
) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    debug_dir = OUTPUT / f"{split}_annotated"
    debug_dir.mkdir(exist_ok=True)
    model_names = {int(key): value for key, value in model.names.items()}
    name_to_id = {value: key for key, value in model_names.items()}
    totals = Counter()
    per_class = defaultdict(Counter)
    frame_records = []

    for frame in frames:
        image = cv2.imread(str(frame["path"]))
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
        results = model.predict(crops, imgsz=224, batch=max(1, len(crops)), device=0, verbose=False) if crops else []
        allowed = ALLOWED[frame["policy"]]
        allowed_ids = [name_to_id[name] for name in allowed]
        for item, result in zip(detected, results):
            probabilities = result.probs.data.cpu().numpy()
            raw_id = int(np.argmax(probabilities))
            allowed_id = max(allowed_ids, key=lambda index: probabilities[index])
            item["raw_class"] = model_names[raw_id]
            item["raw_confidence"] = float(probabilities[raw_id])
            item["allowed_class"] = model_names[allowed_id]
            item["allowed_confidence"] = float(probabilities[allowed_id])
            allowed_mass = float(sum(probabilities[index] for index in allowed_ids))
            item["pairwise_confidence"] = (
                item["allowed_confidence"] / allowed_mass if allowed_mass > 0 else 0.0
            )
            item["accepted"] = (
                item["pairwise_confidence"] >= PAIRWISE_CONFIDENCE_THRESHOLD
            )

        matches, misses, extras = match_targets(frame["targets"], detected)
        totals.update(
            images=1,
            targets=len(frame["targets"]),
            camera_blocked=int(blocked),
            proposal_tp=len(matches),
            proposal_fn=len(misses),
            proposal_fp=len(extras),
        )
        for target_index, proposal_index in matches:
            target, item = frame["targets"][target_index], detected[proposal_index]
            raw_correct = item["raw_class"] == target["class"]
            allowed_correct = item["allowed_class"] == target["class"]
            accepted_correct = item["accepted"] and allowed_correct
            totals.update(
                raw_correct=int(raw_correct),
                allowed_correct=int(allowed_correct),
                accepted=int(item["accepted"]),
                accepted_correct=int(accepted_correct),
            )
            per_class[target["class"]].update(
                matched=1,
                raw_correct=int(raw_correct),
                allowed_correct=int(allowed_correct),
                accepted_correct=int(accepted_correct),
            )
            item["matched_class"] = target["class"]
        for target_index in misses:
            per_class[frame["targets"][target_index]["class"]].update(missed=1)
        for proposal_index in extras:
            if detected[proposal_index].get("accepted"):
                totals.update(accepted_false_positive=1)

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
        for target in frame["targets"]:
            cv2.circle(annotated, tuple(int(round(v)) for v in target["center"]), 8, (255, 0, 255), 2)
        for x0, y0, x1, y1 in bins:
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
        for item in detected:
            x0, y0, x1, y1 = item["box"]
            colour = (0, 200, 0) if item.get("accepted") else (0, 120, 255)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), colour, 2)
            label = f"{item.get('allowed_class','?')} pair={item.get('pairwise_confidence',0):.2f}"
            cv2.putText(annotated, label, (x0, max(18, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        debug_path = debug_dir / f"{frame['session']}__{frame['frame'][:-4]}.jpg"
        cv2.imwrite(str(debug_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frame_records.append(
            {
                "session": frame["session"],
                "frame": frame["frame"],
                "policy": frame["policy"],
                "camera_blocked": blocked,
                "occluder_box": occluder_box,
                "occluder_area": occluder_area,
                "targets": frame["targets"],
                "proposals": detected,
                "missed_target_indices": misses,
                "extra_proposal_indices": extras,
                "annotated": str(debug_path.relative_to(ROOT)),
            }
        )

    matched = totals["proposal_tp"]
    target_count = totals["targets"]
    report = {
        "split": split,
        "confidence_type": "probability normalized within the two allowed task classes",
        "pairwise_confidence_threshold": PAIRWISE_CONFIDENCE_THRESHOLD,
        "frames": totals["images"],
        "targets": target_count,
        "proposal_recall": matched / max(1, target_count),
        "false_proposals_per_frame": totals["proposal_fp"] / max(1, totals["images"]),
        "raw_six_class_accuracy_on_matched": totals["raw_correct"] / max(1, matched),
        "task_allowlisted_accuracy_on_matched": totals["allowed_correct"] / max(1, matched),
        "end_to_end_allowlisted_correct_recall": totals["allowed_correct"] / max(1, target_count),
        "end_to_end_accepted_correct_recall": totals["accepted_correct"] / max(1, target_count),
        "accepted_false_positives": totals["accepted_false_positive"],
        "counts": dict(totals),
        "by_class": {key: dict(value) for key, value in sorted(per_class.items())},
        "frame_records": frame_records,
    }
    return report


def main() -> None:
    frames_by_key = load_frames()
    frames = list(frames_by_key.values())
    camera = json.loads(CAMERA_FILE.read_text(encoding="utf-8"))["table"]
    corners = project_table(
        [(x, y, -0.0051) for x, y in ((0.27, -0.28), (0.50, -0.28), (0.50, 0.28), (0.27, 0.28))],
        camera,
    )
    workspace_polygon = np.round(corners).astype(np.int32)

    tuning_frames = [frame for frame in frames if frame["split"] in {"train", "val"}]
    candidates = []
    for values in itertools.product(
        (90, 110, 130, 150),
        (45, 65, 85),
        (8, 15, 25),
        (900, 1500, 2400),
        (75, 95, 120),
    ):
        config = ProposalConfig(*values)
        metrics = proposal_metrics(tuning_frames, config, workspace_polygon)
        candidates.append((metrics["f1"], metrics["recall"], -metrics["false_proposals_per_image"], config, metrics))
    candidates.sort(key=lambda item: item[:3], reverse=True)
    _, _, _, best_config, tuning_metrics = candidates[0]

    model = YOLO(str(MODEL))
    validation_report = evaluate(
        [frame for frame in frames if frame["split"] == "val"],
        best_config,
        workspace_polygon,
        model,
        "validation",
    )
    test_report = evaluate(
        [frame for frame in frames if frame["split"] == "test"],
        best_config,
        workspace_polygon,
        model,
        "test",
    )
    report = {
        "status": "complete",
        "proposal_parameters_selected_without_test_split": True,
        "bins_used_as_classifier_classes": False,
        "workspace_polygon_px": workspace_polygon.tolist(),
        "best_config": best_config.__dict__,
        "tuning_metrics_train_plus_validation": tuning_metrics,
        "validation": validation_report,
        "test": test_report,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "full_frame_report.json").write_text(json.dumps(report, indent=2) + "\n")
    compact = dict(report)
    compact["validation"] = {key: value for key, value in validation_report.items() if key != "frame_records"}
    compact["test"] = {key: value for key, value in test_report.items() if key != "frame_records"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
