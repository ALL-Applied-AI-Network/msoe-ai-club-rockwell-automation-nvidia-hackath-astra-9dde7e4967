"""Trained-model proposals with calibrated, fail-closed metric verification."""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, replace

import cv2
import numpy as np
import requests

from calibration.geometry import SURFACE_Z, load, pixel_to_base, project
from pill_policy.bin_profiles import (
    BIN_PROFILES,
    DEFAULT_BIN_PROFILE,
    classify_bin_profile,
    metric_bbox_span_mm,
)
from pill_policy.config import (
    CLASS_LABELS,
    CLASS_TO_BIN,
    BIN_VISUAL_MIDPOINT_M,
    DETECTOR_PORT,
    GEOMETRY,
    MAX_TASK_OBJECTS,
    MIN_SEPARATION_MM,
    MODEL_CLASSES,
    OBJECT_HEIGHT_M,
    VERIFY_CLASS_BY_GEOMETRY,
)
from pill_policy.instruction import SortTask


class PerceptionError(RuntimeError):
    pass


class CameraBlockedError(PerceptionError):
    """The full-frame source scan is unsafe because the robot occludes it."""


@dataclass(frozen=True)
class SortTarget:
    sort_class: str
    label: str
    object_xy_mm: tuple[float, float]
    object_yaw_deg: float
    bin_color: str
    bin_xy_mm: tuple[float, float]
    bin_profile: str
    bin_footprint_mm: float
    bin_body_height_mm: float
    bin_release_z_mm: float
    visual_size_px: float
    visual_span_px: float
    object_length_mm: float
    object_width_mm: float
    model_confidence: float
    grasp_feature: str
    relocalization_mm: float = 0.0
    relocalization_yaw_deg: float = 0.0
    bin_relocalization_mm: float = 0.0


class TrainedDetectorClient:
    def __init__(self, url: str | None = None):
        self.url = url or f"http://127.0.0.1:{DETECTOR_PORT}"
        self.cameras = load()
        self.active_class_to_bin = dict(CLASS_TO_BIN)
        self.active_bin_profiles: dict[str, str] = {}

    @staticmethod
    def mock_scene(task: SortTask) -> dict:
        names = list(task.class_to_bin)
        objects = []
        centers = ([0.39, 0.62], [0.47, 0.68], [0.55, 0.63], [0.61, 0.70])
        classes = (names[0], names[0], names[1], names[1])
        for index, (sort_class, center) in enumerate(zip(classes, centers)):
            objects.append(
                {
                    "label": f"{CLASS_LABELS[sort_class]} {index}",
                    "sort_class": sort_class,
                    "center": center,
                    "grasp_center": center,
                    "axis_endpoints": [
                        [center[0] - 0.015, center[1]],
                        [center[0] + 0.015, center[1]],
                    ],
                    "graspable": True,
                    "object_length_mm": sum(GEOMETRY[sort_class]["length"]) / 2,
                    "object_width_mm": sum(GEOMETRY[sort_class]["width"]) / 2,
                    "visual_size_px": 18.0,
                    "visual_span_px": 40.0,
                    "model_confidence": 1.0,
                    "grasp_feature": "body_midpoint",
                }
            )
        colors = list(dict.fromkeys(task.class_to_bin.values()))
        profile_name = "old" if "yellow" in colors else DEFAULT_BIN_PROFILE
        profile = BIN_PROFILES[profile_name]
        bin_centers = ([0.35, 0.76], [0.45, 0.76])
        return {
            "status": "PASS",
            "objects": objects,
            "bins": [
                {
                    "color": color,
                    "center": center,
                    "profile": profile_name,
                    "footprint_mm": profile["footprint_mm"],
                    "body_height_m": profile["body_height_m"],
                    "visual_midpoint_m": profile["visual_midpoint_m"],
                    "release_z_mm": profile["release_z_mm"],
                }
                for color, center in zip(colors, bin_centers)
            ],
            "verification": "mock dynamic inventory",
        }

    def _object_crop(self, image_rgb: np.ndarray):
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        cam = self.cameras["table"]
        corners = project(
            cam,
            [[x, y, SURFACE_Z] for x in (0.27, 0.50) for y in (-0.28, 0.28)],
        )
        scale = np.asarray([width / cam["size"][0], height / cam["size"][1]])
        corners *= scale
        if not np.all(np.isfinite(corners)):
            raise PerceptionError("Calibrated object workspace is outside the image")
        margin = int(round(30 * min(scale)))
        left = max(0, int(np.floor(corners[:, 0].min())) - margin)
        right = min(width, int(np.ceil(corners[:, 0].max())) + margin)
        top = max(0, int(np.floor(corners[:, 1].min())) - margin)
        bottom = min(height, int(np.ceil(corners[:, 1].max())) + margin)
        if right - left < 100 or bottom - top < 100:
            raise PerceptionError("Calibrated object workspace crop is empty")
        return image[top:bottom, left:right], (left, top, right, bottom)

    @staticmethod
    def _map_crop_proposal(item: dict, crop_box, full_shape) -> dict:
        left, top, right, bottom = crop_box
        crop_width, crop_height = right - left, bottom - top
        full_height, full_width = full_shape[:2]

        def point(value):
            return [
                (left + float(value[0]) * crop_width) / full_width,
                (top + float(value[1]) * crop_height) / full_height,
            ]

        mapped = dict(item)
        mapped["center"] = point(item["center"])
        x0, y0, x1, y1 = item["bbox"]
        mapped["bbox"] = point([x0, y0]) + point([x1, y1])
        return mapped

    def _detector_proposals(
        self, image_rgb: np.ndarray, *, enforce_camera_clear: bool = True
    ) -> list[dict]:
        # The detector was trained on complete calibrated 1920x1080 scenes.
        # Preserve that scale/context, then gate returned centres to the
        # calibrated loose-object workspace. Cropping before inference made
        # small objects look like a different size class.
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        _, (left, top, right, bottom) = self._object_crop(image)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise PerceptionError("Could not encode the detector image")
        try:
            response = requests.post(
                self.url + "/infer",
                json={
                    "image_jpeg_base64": base64.b64encode(encoded).decode("ascii"),
                    "enforce_camera_clear": enforce_camera_clear,
                    "allowed_classes": list(MODEL_CLASSES),
                },
                timeout=30,
            )
            payload = response.json()
            response.raise_for_status()
        except (requests.RequestException, ValueError) as exc:
            raise PerceptionError(f"Trained detector service failed: {exc}") from exc
        if payload.get("status") == "BLOCKED":
            raise CameraBlockedError(payload.get("error", "Camera workspace is blocked"))
        if payload.get("status") != "PASS":
            raise PerceptionError(payload.get("error", "Trained detector rejected image"))
        proposals = []
        # Metric geometry remains authoritative, so retain low-margin
        # classifier proposals as candidates. They are never accepted directly.
        classified = list(payload.get("objects", [])) + list(
            payload.get("ambiguous_objects", [])
        )
        for item in classified:
            if item.get("sort_class") not in MODEL_CLASSES:
                continue
            center = np.asarray(item.get("center", []), dtype=float)
            if center.shape != (2,):
                continue
            center_px = center * [width, height]
            if not (left <= center_px[0] <= right and top <= center_px[1] <= bottom):
                continue
            proposals.append(dict(item))
        if len(proposals) > MAX_TASK_OBJECTS * 3:
            raise PerceptionError("Detector returned an ambiguous number of proposals")
        return proposals

    def _local_pill_proposals(
        self, image_rgb: np.ndarray, *, calibrated_workspace: bool = True
    ) -> list[dict]:
        """Find small chromatic objects without trusting a model class label.

        The real pills supplied for the CSI cell are strongly coloured while the
        bed and its shadows have low saturation.  This path supplies candidate
        boxes only; calibrated metric geometry in ``_fit_once`` remains the
        authority for round-versus-long classification.
        """

        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        if calibrated_workspace:
            crop, (left, top, right, bottom) = self._object_crop(image)
        else:  # Offline replay of phone photographs; never used for robot XY.
            crop = image
            left, top, right, bottom = 0, 0, width, height

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        # Union the observed pill colours into one proposal mask.  Colour never
        # selects a bin or class; it only rejects the low-saturation bed and its
        # multiple shadows before calibrated shape verification.
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 55, 35), (15, 255, 255)),
            cv2.inRange(hsv, (160, 55, 35), (179, 255, 255)),
        )
        yellow = cv2.inRange(hsv, (14, 70, 40), (52, 255, 255))
        mask = cv2.bitwise_or(red, yellow)
        kernel_size = max(3, int(round(min(crop.shape[:2]) / 430.0)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        crop_area = float(crop.shape[0] * crop.shape[1])
        min_area = max(20, int(round(crop_area * 0.000025)))
        max_area = max(min_area + 1, int(round(crop_area * 0.005)))
        max_span = max(75, int(round(min(crop.shape[:2]) * 0.12)))
        proposals = []
        for index in range(1, count):
            x, y, box_width, box_height, area = [
                int(value) for value in stats[index]
            ]
            if not min_area <= area <= max_area:
                continue
            if min(box_width, box_height) < 4 or max(box_width, box_height) > max_span:
                continue
            fill = area / max(float(box_width * box_height), 1.0)
            if fill < 0.28:
                continue
            points = cv2.findNonZero((labels == index).astype(np.uint8))
            if points is None or len(points) < 20:
                continue
            rect_width, rect_height = cv2.minAreaRect(points)[1]
            short = max(min(rect_width, rect_height), 1.0)
            shape_aspect = max(rect_width, rect_height) / short
            if not 0.80 <= shape_aspect <= 4.50:
                continue

            center_x, center_y = centers[index]
            # This provisional label only supplies an object-height estimate.
            # It is overwritten by calibrated geometry before planning.
            provisional = "long_pill" if shape_aspect >= 1.45 else "round_pill"
            proposals.append(
                {
                    "sort_class": provisional,
                    "confidence": 0.0,
                    "center": [
                        (left + float(center_x)) / width,
                        (top + float(center_y)) / height,
                    ],
                    "bbox": [
                        (left + x) / width,
                        (top + y) / height,
                        (left + x + box_width) / width,
                        (top + y + box_height) / height,
                    ],
                    "shape_aspect": float(shape_aspect),
                    "source": "local chromatic component",
                }
            )
        if len(proposals) > MAX_TASK_OBJECTS * 3:
            raise PerceptionError("Local scan returned an ambiguous number of proposals")
        return proposals

    @staticmethod
    def _component_near(mask: np.ndarray, target: np.ndarray, min_area: int = 20):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            distance = float(np.linalg.norm(centers[index] - target))
            candidates.append((distance - min(area, 1000) * 0.002, distance, index))
        if not candidates:
            return None
        _, distance, index = min(candidates)
        if distance > 0.35 * max(mask.shape):
            return None
        return labels == index

    def _metric_distance_mm(self, p0, p1, image_shape, object_height) -> float:
        height, width = image_shape[:2]
        native_width, native_height = self.cameras["table"]["size"]
        scale = np.asarray([native_width / width, native_height / height])
        a = pixel_to_base(
            self.cameras["table"], np.asarray(p0) * scale, z=SURFACE_Z + object_height
        )
        b = pixel_to_base(
            self.cameras["table"], np.asarray(p1) * scale, z=SURFACE_Z + object_height
        )
        return float(np.linalg.norm((b - a)[:2]) * 1000.0)

    @staticmethod
    def _classify_geometry(length_mm: float, width_mm: float) -> str:
        aspect = length_mm / max(width_mm, 1e-6)
        matches = []
        for sort_class, limits in GEOMETRY.items():
            if (
                limits["length"][0] <= length_mm <= limits["length"][1]
                and limits["width"][0] <= width_mm <= limits["width"][1]
                and limits["aspect"][0] <= aspect <= limits["aspect"][1]
            ):
                matches.append(sort_class)
        if len(matches) != 1:
            raise PerceptionError(
                f"Metric geometry is ambiguous: {length_mm:.1f} x {width_mm:.1f} mm"
            )
        return matches[0]

    def _fit_once(self, image_rgb, proposal, threshold: float) -> dict:
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        x0, y0, x1, y1 = np.asarray(proposal["bbox"], float) * [width, height, width, height]
        center = np.asarray(proposal["center"], float) * [width, height]
        pad_x = max(10.0, 0.25 * (x1 - x0))
        pad_y = max(10.0, 0.25 * (y1 - y0))
        left = max(0, int(np.floor(x0 - pad_x)))
        right = min(width, int(np.ceil(x1 + pad_x)))
        top = max(0, int(np.floor(y0 - pad_y)))
        bottom = min(height, int(np.ceil(y1 + pad_y)))
        patch = image[top:bottom, left:right]
        if min(patch.shape[:2]) < 12:
            raise PerceptionError("Detector proposal crop is too small")
        lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
        border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab - background, axis=2)
        mask = (distance >= threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        component = self._component_near(mask, center - [left, top])
        if component is None:
            raise PerceptionError("No pixel-supported object inside model proposal")
        ys, xs = np.nonzero(component)
        points = np.column_stack((xs + left, ys + top)).astype(np.float64)
        if len(points) < 20:
            raise PerceptionError("Object silhouette is too small")
        mean = points.mean(axis=0)
        _, singular, vectors = np.linalg.svd(points - mean, full_matrices=False)
        axis = vectors[0]
        cross = vectors[1]
        along_values = (points - mean) @ axis
        cross_values = (points - mean) @ cross
        along_low, along_high = np.percentile(along_values, [2.0, 98.0])
        cross_low, cross_high = np.percentile(cross_values, [5.0, 95.0])
        length_px = float(along_high - along_low)
        width_px = float(cross_high - cross_low)
        if width_px > length_px:
            axis, cross = cross, axis
            length_px, width_px = width_px, length_px
            along_values = (points - mean) @ axis
            cross_values = (points - mean) @ cross
            along_low, along_high = np.percentile(along_values, [2.0, 98.0])
            cross_low, cross_high = np.percentile(cross_values, [5.0, 95.0])
        p0 = mean + axis * along_low
        p1 = mean + axis * along_high
        q0 = mean + cross * cross_low
        q1 = mean + cross * cross_high
        # Proposal labels are not authoritative.  A common pill-plane estimate
        # avoids making the metric measurement depend on a possibly wrong model
        # class; the sub-millimetre height difference is inside calibration
        # tolerance and the final disjoint geometry gates remain fail-closed.
        object_height = float(np.mean(tuple(OBJECT_HEIGHT_M.values())))
        length_mm = self._metric_distance_mm(p0, p1, image.shape, object_height)
        width_mm = self._metric_distance_mm(q0, q1, image.shape, object_height)
        if VERIFY_CLASS_BY_GEOMETRY:
            measured_class = self._classify_geometry(length_mm, width_mm)
        else:
            measured_class = proposal["sort_class"]
            limits = GEOMETRY[measured_class]
            aspect = length_mm / max(width_mm, 1e-6)
            if not (
                limits["length"][0] <= length_mm <= limits["length"][1]
                and limits["width"][0] <= width_mm <= limits["width"][1]
                and limits["aspect"][0] <= aspect <= limits["aspect"][1]
            ):
                raise PerceptionError(
                    f"{measured_class} proposal failed its metric family gate: "
                    f"{length_mm:.1f} x {width_mm:.1f} mm"
                )
        return {
            "center_px": mean,
            "axis_endpoints_px": np.stack((p0, p1)),
            "length_px": length_px,
            "width_px": width_px,
            "length_mm": length_mm,
            "width_mm": width_mm,
            "sort_class": measured_class,
        }

    def _verify_proposal(self, image_rgb, proposal) -> dict:
        fits = [self._fit_once(image_rgb, proposal, value) for value in (16.0, 22.0)]
        first, second = fits
        if first["sort_class"] != second["sort_class"]:
            raise PerceptionError("Object class changed across segmentation thresholds")
        if np.linalg.norm(first["center_px"] - second["center_px"]) > 6.0:
            raise PerceptionError("Object centre is unstable across segmentation thresholds")
        if abs(first["length_mm"] - second["length_mm"]) > 0.25 * second["length_mm"]:
            raise PerceptionError("Object length is unstable across segmentation thresholds")
        if abs(first["width_mm"] - second["width_mm"]) > 0.30 * second["width_mm"]:
            raise PerceptionError("Object width is unstable across segmentation thresholds")
        height, width = np.asarray(image_rgb).shape[:2]
        center = second["center_px"] / [width, height]
        endpoints = second["axis_endpoints_px"] / [width, height]
        return {
            "label": f"{CLASS_LABELS[second['sort_class']]} (metric verified)",
            "sort_class": second["sort_class"],
            "center": center.tolist(),
            "grasp_center": center.tolist(),
            "axis_endpoints": endpoints.tolist(),
            "graspable": True,
            "object_length_mm": second["length_mm"],
            "object_width_mm": second["width_mm"],
            "visual_size_px": second["width_px"],
            "visual_span_px": second["length_px"],
            "model_confidence": float(proposal.get("confidence", 0.0)),
            "grasp_feature": "body_midpoint",
            "source": str(proposal.get("source", "trained six-class detector"))
            + " + calibrated metric verification",
        }

    def detect_objects(
        self,
        image_rgb: np.ndarray,
        *,
        allow_empty: bool,
        enforce_camera_clear: bool = True,
    ) -> list[dict]:
        local_proposals = self._local_pill_proposals(image_rgb)
        detector_error = None
        try:
            trained_proposals = self._detector_proposals(
                image_rgb, enforce_camera_clear=enforce_camera_clear
            )
        except CameraBlockedError:
            raise
        except PerceptionError as exc:
            detector_error = exc
            trained_proposals = []
        proposals = local_proposals + trained_proposals
        verified = []
        for proposal in proposals:
            try:
                candidate = self._verify_proposal(image_rgb, proposal)
            except PerceptionError:
                continue
            center = np.asarray(candidate["center"])
            if any(np.linalg.norm(center - np.asarray(old["center"])) < 0.010 for old in verified):
                continue
            verified.append(candidate)
        if len(verified) > MAX_TASK_OBJECTS:
            raise PerceptionError(f"Verified inventory exceeds safety ceiling: {len(verified)}")
        if not verified and not allow_empty:
            if detector_error is not None and not local_proposals:
                raise PerceptionError(
                    f"No local proposals and trained detector failed: {detector_error}"
                ) from detector_error
            raise PerceptionError("No task-relevant objects passed model and metric verification")
        return verified

    @staticmethod
    def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
        if color == "red":
            return cv2.bitwise_or(
                cv2.inRange(hsv, (0, 70, 40), (13, 255, 255)),
                cv2.inRange(hsv, (165, 70, 35), (179, 255, 255)),
            )
        if color == "blue":
            return cv2.inRange(hsv, (90, 55, 35), (135, 255, 255))
        if color == "yellow":
            return cv2.inRange(hsv, (15, 60, 50), (48, 255, 255))
        if color == "green":
            return cv2.inRange(hsv, (38, 50, 35), (92, 255, 255))
        raise ValueError(color)

    def _detect_bins(self, image_rgb: np.ndarray) -> list[dict]:
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        _, (workspace_left, workspace_top, workspace_right, workspace_bottom) = (
            self._object_crop(image)
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        kernel = np.ones((3, 3), np.uint8)
        bins = []
        for color in dict.fromkeys(self.active_class_to_bin.values()):
            mask = self._color_mask(hsv, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            candidates = []
            for index in range(1, count):
                x, y, box_width, box_height, area = [int(value) for value in stats[index]]
                ratio = box_width / max(box_height, 1)
                center_x = x + box_width / 2
                center_y = y + box_height / 2
                in_collection_workspace = (
                    workspace_left <= center_x <= workspace_right
                    and workspace_top <= center_y <= workspace_bottom
                )
                if (
                    in_collection_workspace
                    and 600 <= area <= 30000
                    and box_width >= 20
                    and box_height >= 20
                    and 0.4 <= ratio <= 2.5
                ):
                    candidates.append((area, x, y, box_width, box_height))
            if not candidates:
                raise PerceptionError(f"No pixel-verified {color} bin")
            area, x, y, box_width, box_height = max(candidates)
            bbox_pixels = (x, y, x + box_width, y + box_height)
            _, _, metric_span_mm = metric_bbox_span_mm(
                bbox_pixels, image.shape, self.cameras["table"]
            )
            try:
                profile = classify_bin_profile(
                    metric_span_mm, self.active_bin_profiles.get(color)
                )
            except ValueError as exc:
                raise PerceptionError(f"Unsafe {color} bin geometry: {exc}") from exc
            self.active_bin_profiles[color] = profile["name"]
            bins.append(
                {
                    "color": color,
                    "center": [(x + box_width / 2) / width, (y + box_height / 2) / height],
                    "bbox": [
                        x / width,
                        y / height,
                        (x + box_width) / width,
                        (y + box_height) / height,
                    ],
                    "pixel_area": area,
                    "metric_span_mm": metric_span_mm,
                    "profile": profile["name"],
                    "footprint_mm": profile["footprint_mm"],
                    "body_height_m": profile["body_height_m"],
                    "visual_midpoint_m": profile["visual_midpoint_m"],
                    "release_z_mm": profile["release_z_mm"],
                }
            )
        return bins

    @staticmethod
    def _point(item: dict, key: str) -> np.ndarray:
        point = np.asarray(item.get(key, []), dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise PerceptionError(f"Missing {key} in detection")
        if np.any(point < 0.0) or np.any(point > 1.0):
            raise PerceptionError(f"Detection {key} is outside the image")
        return point

    @staticmethod
    def _canonical_axis_yaw(yaw: float) -> float:
        return float((yaw + 90.0) % 180.0 - 90.0)

    @staticmethod
    def _axis_angle_error(a: float, b: float) -> float:
        return float(abs((a - b + 90.0) % 180.0 - 90.0))

    @staticmethod
    def _validate_separation(targets: list[SortTarget]) -> None:
        for index, left in enumerate(targets):
            for right in targets[index + 1 :]:
                distance = float(
                    np.linalg.norm(np.asarray(left.object_xy_mm) - np.asarray(right.object_xy_mm))
                )
                if distance < MIN_SEPARATION_MM:
                    raise PerceptionError(
                        f"Objects overlap or are too close for a safe grasp: {distance:.1f} mm"
                    )

    def infer(self, image_rgb: np.ndarray, task: SortTask) -> dict:
        self.active_bin_profiles = {}
        self.active_class_to_bin = dict(task.class_to_bin)
        return {
            "status": "PASS",
            "objects": self.detect_objects(image_rgb, allow_empty=False),
            "bins": self._detect_bins(image_rgb),
            "verification": "trained detector + calibrated metric geometry",
        }

    def to_targets(self, scene: dict, task: SortTask) -> list[SortTarget]:
        bins_by_color = {item["color"]: item for item in scene.get("bins", [])}
        missing = set(task.class_to_bin.values()) - set(bins_by_color)
        if missing:
            raise PerceptionError("Missing destination bins: " + ", ".join(sorted(missing)))
        objects = [item for item in scene.get("objects", []) if item.get("sort_class") in task.class_to_bin]
        if not objects:
            raise PerceptionError("No task-relevant objects detected")
        if len(objects) > MAX_TASK_OBJECTS:
            raise PerceptionError("Dynamic inventory exceeds its safety ceiling")
        cam = self.cameras["table"]
        native_width, native_height = cam["size"]

        def base(point, height_above_table):
            return pixel_to_base(
                cam,
                [point[0] * native_width, point[1] * native_height],
                z=SURFACE_Z + height_above_table,
            )

        targets = []
        for item in objects:
            sort_class = item["sort_class"]
            object_height = OBJECT_HEIGHT_M[sort_class]
            object_xyz = base(self._point(item, "grasp_center"), object_height)
            endpoints = item.get("axis_endpoints") or []
            yaw = 0.0
            # A round pill has no physically meaningful long-axis yaw.  PCA can
            # rotate arbitrarily between otherwise identical frames, so keep a
            # canonical zero yaw.  Long pills retain measured-axis alignment.
            if sort_class != "round_pill" and len(endpoints) == 2:
                p0 = base(np.asarray(endpoints[0]), object_height)
                p1 = base(np.asarray(endpoints[1]), object_height)
                delta = p1[:2] - p0[:2]
                yaw = self._canonical_axis_yaw(float(np.degrees(np.arctan2(delta[1], delta[0]))))
            color = task.class_to_bin[sort_class]
            bin_item = bins_by_color[color]
            bin_xyz = base(
                self._point(bin_item, "center"),
                float(bin_item["visual_midpoint_m"]),
            )
            targets.append(
                SortTarget(
                    sort_class=sort_class,
                    label=str(item.get("label", CLASS_LABELS[sort_class])),
                    object_xy_mm=(float(object_xyz[0] * 1000), float(object_xyz[1] * 1000)),
                    object_yaw_deg=yaw,
                    bin_color=color,
                    bin_xy_mm=(float(bin_xyz[0] * 1000), float(bin_xyz[1] * 1000)),
                    bin_profile=str(bin_item["profile"]),
                    bin_footprint_mm=float(bin_item["footprint_mm"]),
                    bin_body_height_mm=float(bin_item["body_height_m"] * 1000.0),
                    bin_release_z_mm=float(bin_item["release_z_mm"]),
                    visual_size_px=float(item.get("visual_size_px", 0.0)),
                    visual_span_px=float(item.get("visual_span_px", 0.0)),
                    object_length_mm=float(item.get("object_length_mm", 0.0)),
                    object_width_mm=float(item.get("object_width_mm", 0.0)),
                    model_confidence=float(item.get("model_confidence", 0.0)),
                    grasp_feature=str(item.get("grasp_feature", "body_midpoint")),
                )
            )
        order = {name: index for index, name in enumerate(task.class_to_bin)}
        targets.sort(key=lambda item: (order[item.sort_class], np.hypot(*item.object_xy_mm)))
        self._validate_separation(targets)
        return targets

    def relocalize_targets(self, image_rgb: np.ndarray, old_targets: list[SortTarget]) -> list[SortTarget]:
        if not old_targets:
            return []
        objects = self.detect_objects(
            image_rgb, allow_empty=True, enforce_camera_clear=False
        )
        if not objects:
            raise PerceptionError("No pixel-supported remaining targets")
        bins = self._detect_bins(image_rgb)
        current = self.to_targets(
            {"objects": objects, "bins": bins},
            SortTask("relocalization", dict(self.active_class_to_bin)),
        )
        unused = list(current)
        refreshed = []
        for old in old_targets:
            candidates = [item for item in unused if item.sort_class == old.sort_class]
            if not candidates:
                raise PerceptionError(f"No pixel-supported {old.sort_class} near previous target")
            new = min(
                candidates,
                key=lambda item: np.linalg.norm(
                    np.asarray(item.object_xy_mm) - np.asarray(old.object_xy_mm)
                ),
            )
            shift = float(np.linalg.norm(np.asarray(new.object_xy_mm) - np.asarray(old.object_xy_mm)))
            if shift > 12.0:
                raise PerceptionError(f"{old.label} moved {shift:.1f} mm")
            if new.bin_profile != old.bin_profile:
                raise PerceptionError(
                    f"{old.bin_color} bin profile changed from {old.bin_profile} "
                    f"to {new.bin_profile}"
                )
            bin_shift = float(np.linalg.norm(np.asarray(new.bin_xy_mm) - np.asarray(old.bin_xy_mm)))
            if bin_shift > 25.0:
                raise PerceptionError(f"{old.bin_color} bin moved {bin_shift:.1f} mm")
            unused.remove(new)
            refreshed.append(
                replace(
                    new,
                    label=old.label,
                    bin_xy_mm=old.bin_xy_mm,
                    bin_profile=old.bin_profile,
                    bin_footprint_mm=old.bin_footprint_mm,
                    bin_body_height_mm=old.bin_body_height_mm,
                    bin_release_z_mm=old.bin_release_z_mm,
                    relocalization_mm=shift,
                    relocalization_yaw_deg=self._axis_angle_error(
                        new.object_yaw_deg, old.object_yaw_deg
                    ),
                    bin_relocalization_mm=bin_shift,
                )
            )
        self._validate_separation(refreshed)
        return refreshed

    def target_is_present(self, image_rgb: np.ndarray, target: SortTarget) -> bool:
        # Pickup verification runs from the guarded above-bin retreat pose.
        # Keep the object/metric checks, but do not require the initial-scan
        # camera-clear pose for this one local verification step.
        objects = self.detect_objects(
            image_rgb, allow_empty=True, enforce_camera_clear=False
        )
        if not objects:
            return False
        cam = self.cameras["table"]
        native_width, native_height = cam["size"]
        for item in objects:
            if item["sort_class"] != target.sort_class:
                continue
            point = self._point(item, "grasp_center")
            xyz = pixel_to_base(
                cam,
                [point[0] * native_width, point[1] * native_height],
                z=SURFACE_Z + OBJECT_HEIGHT_M[target.sort_class],
            )
            distance = float(
                np.linalg.norm(np.asarray(xyz[:2]) * 1000.0 - np.asarray(target.object_xy_mm))
            )
            if distance <= 12.0:
                return True
        return False

    def residual_targets(self, image_rgb: np.ndarray, task: SortTask) -> list[SortTarget]:
        self.active_class_to_bin = dict(task.class_to_bin)
        objects = self.detect_objects(image_rgb, allow_empty=True)
        if not objects:
            return []
        return self.to_targets(
            {"objects": objects, "bins": self._detect_bins(image_rgb)}, task
        )
