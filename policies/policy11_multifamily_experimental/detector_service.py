#!/usr/bin/env python3
"""Full-frame proposals plus Astra's six-class crop classifier.

The service is task-gated by ``MODEL_CLASSES`` from the active policy. It
never decides robot coordinates: the policy's calibrated metric verifier
remains the authority for object class, centre, yaw, and grasp geometry.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from calibration.geometry import SURFACE_Z, load, project
from family_policy.config import CHECKPOINT, DETECTOR_PORT, MODEL_CLASSES, POLICY_NUMBER


HOST = "127.0.0.1"
PAIRWISE_CONFIDENCE = float(os.environ.get("ASTRA_PAIRWISE_CONF", "0.75"))
MODEL_LOCK = threading.Lock()
CHECKPOINT_PATH = Path(CHECKPOINT)
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"Missing trained checkpoint: {CHECKPOINT_PATH}")
MODEL = YOLO(str(CHECKPOINT_PATH))
MODEL_NAMES = {int(key): str(value) for key, value in MODEL.names.items()}
NAME_TO_ID = {value: key for key, value in MODEL_NAMES.items()}
missing_classes = set(MODEL_CLASSES) - set(NAME_TO_ID)
if missing_classes:
    raise RuntimeError(f"Checkpoint is missing policy classes: {sorted(missing_classes)}")
CHECKPOINT_SHA256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
CAMERAS = load()


def _workspace_polygon(image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    camera = CAMERAS["table"]
    corners = project(
        camera,
        [
            [x, y, SURFACE_Z]
            for x, y in ((0.27, -0.28), (0.50, -0.28), (0.50, 0.28), (0.27, 0.28))
        ],
    )
    native_width, native_height = camera["size"]
    corners *= [width / native_width, height / native_height]
    if not np.all(np.isfinite(corners)):
        raise ValueError("Calibrated table workspace is outside the image")
    return np.round(corners).astype(np.int32)


def _bin_boxes(hsv: np.ndarray, workspace: np.ndarray) -> list[tuple[int, int, int, int]]:
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.where((saturation >= 75) & (value >= 55), 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, workspace)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    boxes = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(contour)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if area >= 500 and box_width >= 28 and box_height >= 28:
            margin = 12
            boxes.append(
                (x - margin, y - margin, x + box_width + margin, y + box_height + margin)
            )
    return boxes


def _camera_occlusion(
    image: np.ndarray, workspace_polygon: np.ndarray
) -> tuple[bool, tuple[int, int, int, int] | None, float]:
    """Detect a large robot component bridging into the loose-object ROI."""
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
        bridges = y <= top_y - 70 and y + box_height >= top_y - 8
        if area >= 1800 and bridges and area > best_area:
            best_area = area
            best_box = (x, y, x + box_width, y + box_height)
    return best_box is not None, best_box, best_area


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    width = max(0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = width * height
    if not intersection:
        return 0.0
    return intersection / min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))


def _proposals(image: np.ndarray, workspace_polygon: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    workspace = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(workspace, workspace_polygon, 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins = _bin_boxes(hsv, workspace)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.where(
        (value <= 90) | ((saturation >= 45) & (value >= 45)), 255, 0
    ).astype(np.uint8)
    mask = cv2.bitwise_and(mask, workspace)
    for x0, y0, x1, y1 in bins:
        cv2.rectangle(
            mask,
            (max(0, x0), max(0, y0)),
            (min(width - 1, x1), min(height - 1, y1)),
            0,
            -1,
        )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    found = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        x, y, box_width, box_height = cv2.boundingRect(contour)
        long_side = max(box_width, box_height)
        short_side = min(box_width, box_height)
        if not 8 <= area <= 900 or long_side > 75 or short_side < 3:
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
                "center_px": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                "area": area,
            }
        )
    found.sort(key=lambda item: item["area"], reverse=True)
    kept = []
    for item in found:
        if any(_overlap(item["box"], old["box"]) > 0.55 for old in kept):
            continue
        kept.append(item)
    return sorted(kept, key=lambda item: (item["center_px"][1], item["center_px"][0]))


def _square_crop(
    image: np.ndarray, box: tuple[int, int, int, int], side: int = 96
) -> np.ndarray:
    x0, y0, x1, y1 = box
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right, bottom = left + side, top + side
    result = np.full((side, side, 3), 235, dtype=np.uint8)
    source_x0, source_y0 = max(0, left), max(0, top)
    source_x1, source_y1 = min(image.shape[1], right), min(image.shape[0], bottom)
    result[source_y0 - top : source_y1 - top, source_x0 - left : source_x1 - left] = image[
        source_y0:source_y1, source_x0:source_x1
    ]
    return result


def infer(
    image: np.ndarray,
    *,
    enforce_camera_clear: bool = True,
    allowed_classes: list[str] | tuple[str, ...] | None = None,
) -> dict:
    height, width = image.shape[:2]
    workspace_polygon = _workspace_polygon(image.shape)
    blocked, box, area = _camera_occlusion(image, workspace_polygon)
    if enforce_camera_clear and blocked:
        return {
            "status": "BLOCKED",
            "error": "Robot wrist/gripper overlaps the calibrated source workspace",
            "occluder_box": box,
            "occluder_area": area,
        }

    proposals = _proposals(image, workspace_polygon)
    crops = [_square_crop(image, item["box"]) for item in proposals]
    with MODEL_LOCK:
        results = (
            MODEL.predict(
                crops,
                imgsz=224,
                batch=max(1, len(crops)),
                device=0,
                verbose=False,
            )
            if crops
            else []
        )
    requested_classes = tuple(allowed_classes or MODEL_CLASSES)
    if not requested_classes or not set(requested_classes) <= set(MODEL_CLASSES):
        raise ValueError(f"Unsupported classifier class filter: {requested_classes}")
    allowed_ids = [NAME_TO_ID[name] for name in requested_classes]
    accepted, ambiguous = [], []
    for proposal, result in zip(proposals, results):
        probabilities = result.probs.data.cpu().numpy()
        class_id = max(allowed_ids, key=lambda index: probabilities[index])
        allowed_mass = float(sum(probabilities[index] for index in allowed_ids))
        pairwise = float(probabilities[class_id] / allowed_mass) if allowed_mass > 0 else 0.0
        x0, y0, x1, y1 = proposal["box"]
        item = {
            "sort_class": MODEL_NAMES[class_id],
            "confidence": pairwise,
            "absolute_confidence": float(probabilities[class_id]),
            "center": [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height)],
            "bbox": [x0 / width, y0 / height, x1 / width, y1 / height],
            "source": "astra full-frame proposal + six-class crop classifier",
        }
        (accepted if pairwise >= PAIRWISE_CONFIDENCE else ambiguous).append(item)
    return {
        "status": "PASS",
        "objects": accepted,
        "ambiguous_objects": ambiguous,
        "proposal_count": len(proposals),
        "requires_fallback": bool(ambiguous),
        "allowed_classes": list(requested_classes),
        "pairwise_confidence_threshold": PAIRWISE_CONFIDENCE,
        "camera_occlusion_detected": blocked,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(
            200,
            {
                "status": "PASS",
                "policy": POLICY_NUMBER,
                "service": "full-frame-classifier",
                "classes": list(MODEL_CLASSES),
                "pairwise_confidence_threshold": PAIRWISE_CONFIDENCE,
                "checkpoint_sha256": CHECKPOINT_SHA256,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            raw = base64.b64decode(request["image_jpeg_base64"], validate=True)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("JPEG decode failed")
            self._send(
                200,
                infer(
                    image,
                    enforce_camera_clear=bool(request.get("enforce_camera_clear", True)),
                    allowed_classes=request.get("allowed_classes"),
                ),
            )
        except Exception as exc:  # fail closed at the policy boundary
            self._send(400, {"status": "FAIL", "error": str(exc)})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, DETECTOR_PORT), Handler).serve_forever()
