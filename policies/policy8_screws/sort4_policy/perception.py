"""Cosmos-grounded, locally verified M4/M8 screw perception for Policy 8."""

from __future__ import annotations

import base64
import os
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import requests

from calibration.geometry import SURFACE_Z, load, pixel_to_base, project
from sort4_policy.bin_profiles import (
    BIN_PROFILES,
    DEFAULT_BIN_PROFILE,
    classify_bin_profile,
    metric_bbox_span_mm,
)
from sort4_policy.instruction import CLASS_TO_BIN, Sort4Task


@dataclass(frozen=True)
class SortTarget:
    sort_class: str
    label: str
    object_xy_mm: tuple[float, float]
    object_yaw_deg: float
    bin_color: str
    bin_xy_mm: tuple[float, float]
    bin_profile: str = DEFAULT_BIN_PROFILE
    bin_footprint_mm: float = 90.0
    bin_body_height_mm: float = 60.0
    bin_release_z_mm: float = -265.0
    visual_size_px: float = 0.0
    visual_span_px: float = 0.0
    initial_refinement_mm: float = 0.0
    initial_yaw_refinement_deg: float = 0.0
    relocalization_mm: float = 0.0
    relocalization_yaw_deg: float = 0.0
    bin_relocalization_mm: float = 0.0
    head_width_mm: float = 0.0
    shank_width_mm: float = 0.0
    screw_length_mm: float = 0.0
    cube_side_mm: float = 0.0
    grasp_feature: str = "center"


class PerceptionError(RuntimeError):
    pass


OBJECT_HEIGHT_M = {
    "m4_screw": 0.004,
    "m8_screw": 0.007,
}

# Dimensions supplied by Astra from screw_size.jpeg.  Every screw has a
# 20 mm shaft; the grasp point is the midpoint of that shaft, never the head.
SCREW_GEOMETRY_MM = {
    "m4_screw": {
        "head_diameter": 7.0,
        "head_height": 4.0,
        "shank_diameter": 4.0,
        "shank_length": 20.0,
        "total_length": 24.0,
        "head_range": (4.5, 9.0),
        "shank_range": (2.0, 5.8),
        "length_range": (18.0, 29.0),
    },
    "m8_screw": {
        "head_diameter": 13.0,
        "head_height": 8.0,
        "shank_diameter": 8.0,
        "shank_length": 20.0,
        "total_length": 28.0,
        "head_range": (10.5, 16.5),
        "shank_range": (6.5, 11.0),
        "length_range": (22.0, 34.0),
    },
}
# The real CSI image has a soft, repeatable shadow around black parts.  These
# ranges describe that observed silhouette, while SCREW_GEOMETRY_MM above
# remains the strict physical-dimension contract used by validation/tests.
OBSERVED_SCREW_SILHOUETTE_MM = {
    "m4_screw": {
        "head_range": (4.5, 10.5),
        "shank_range": (2.0, 8.3),
        "length_range": (18.0, 32.0),
    },
    "m8_screw": {
        "head_range": (11.0, 20.0),
        "shank_range": (7.5, 15.0),
        "length_range": (22.0, 41.0),
    },
}
CUBE_SIDE_RANGE_MM = (7.0, 18.5)
# Bins are stationary fixtures for a trial.  Their colored body can be partly
# occluded by the gripper after a drop, which moved the HSV bounding-box centre
# 16--18 mm in the real run even though the bin itself had not moved.  Treat
# that as bounded presence evidence and retain the initially verified centre.
MAX_BIN_EVIDENCE_SHIFT_MM = 25.0
BIN_BODY_HEIGHT_M = BIN_PROFILES[DEFAULT_BIN_PROFILE]["body_height_m"]
BIN_VISUAL_MIDPOINT_M = BIN_BODY_HEIGHT_M / 2.0
# Runtime inventory is dynamic. This is only a fail-closed safety ceiling, not
# an expected count. It can be raised explicitly for an approved placement.
MAX_TASK_OBJECTS = int(os.environ.get("DEMO_MAX_SCREWS", "24"))
MAX_OBJECTS_PER_CLASS = MAX_TASK_OBJECTS
MAX_LOCAL_SCREW_CANDIDATES = MAX_TASK_OBJECTS
LOCAL_SCAN_MATCH_PX = 12.0


class Sort4CosmosClient:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get(
            "COSMOS_DEMO_URL", "http://127.0.0.1:18765"
        )
        self.cameras = load()
        self.active_class_to_bin = dict(CLASS_TO_BIN)
        self.active_bin_profiles: dict[str, str] = {}

    @staticmethod
    def mock_scene(task: Sort4Task) -> dict:
        # All points map inside the smaller guarded robot work envelope.
        objects = [
            ("m4_screw", [0.36, 0.62]),
            ("m4_screw", [0.43, 0.64]),
            ("m4_screw", [0.50, 0.66]),
            ("m8_screw", [0.55, 0.62]),
            ("m8_screw", [0.60, 0.65]),
            ("m8_screw", [0.65, 0.68]),
        ]
        result = []
        for index, (sort_class, center) in enumerate(objects):
            item = {
                "label": f"{sort_class} {index}",
                "sort_class": sort_class,
                "center": center,
                "grasp_center": center,
                "graspable": True,
            }
            if "screw" in sort_class:
                item["axis_endpoints"] = [
                    [center[0] - 0.018, center[1] - 0.006],
                    [center[0] + 0.018, center[1] + 0.006],
                ]
            result.append(item)
        colors = list(dict.fromkeys(task.class_to_bin.values()))
        profile_name = "old" if "yellow" in colors else DEFAULT_BIN_PROFILE
        profile = BIN_PROFILES[profile_name]
        bin_centers = ([0.35, 0.76], [0.45, 0.76])
        return {
            "status": "PASS",
            "objects": result,
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
        }

    def _object_crop(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop the calibrated loose-object area; no object coordinates are used."""
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        cam = self.cameras["table"]
        corners = project(
            cam,
            [
                [x, y, SURFACE_Z]
                for x in (0.29, 0.47)
                for y in (-0.25, 0.25)
            ],
        )
        scale = np.asarray([width / cam["size"][0], height / cam["size"][1]])
        corners = corners * scale
        if not np.all(np.isfinite(corners)):
            raise PerceptionError("The calibrated task workspace is outside the table image")
        margin = int(round(40 * min(scale)))
        left = max(0, int(np.floor(corners[:, 0].min())) - margin)
        right = min(width, int(np.ceil(corners[:, 0].max())) + margin)
        top = max(0, int(np.floor(corners[:, 1].min())) - margin)
        bottom = min(height, int(np.ceil(corners[:, 1].max())) + margin)
        if right - left < 100 or bottom - top < 100:
            raise PerceptionError("The calibrated task workspace crop is empty")
        return image[top:bottom, left:right], (left, top, right, bottom)

    @staticmethod
    def _map_crop_scene(
        scene: dict,
        crop_box: tuple[int, int, int, int],
        full_shape: tuple[int, ...],
    ) -> dict:
        left, top, right, bottom = crop_box
        crop_width, crop_height = right - left, bottom - top
        full_height, full_width = full_shape[:2]

        def point(value):
            return [
                (left + float(value[0]) * crop_width) / full_width,
                (top + float(value[1]) * crop_height) / full_height,
            ]

        objects = []
        for raw in scene.get("objects", []):
            item = dict(raw)
            for key in ("center", "grasp_center"):
                if item.get(key) is not None:
                    item[key] = point(item[key])
            if item.get("bbox") is not None:
                x0, y0, x1, y1 = item["bbox"]
                item["bbox"] = point([x0, y0]) + point([x1, y1])
            if item.get("axis_endpoints") is not None:
                item["axis_endpoints"] = [point(value) for value in item["axis_endpoints"]]
            objects.append(item)
        return {**scene, "objects": objects, "bins": []}

    @staticmethod
    def _component_near(mask: np.ndarray, target: np.ndarray, min_area: int = 12):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            distance = float(np.linalg.norm(centers[index] - target))
            candidates.append((distance - 0.01 * area, distance, index, area))
        if not candidates:
            return None
        _, distance, index, area = min(candidates)
        if distance > max(12.0, 0.35 * min(mask.shape[:2])):
            return None
        return labels == index, area

    def _fit_expected_screw_component_near(
        self,
        mask: np.ndarray,
        target: np.ndarray,
        offset: np.ndarray,
        image_shape: tuple[int, ...],
        expected_class: str,
        min_area: int = 12,
    ) -> dict:
        """Fit the nearest plausible screw, ignoring seams and robot edges.

        Relocalization crops can contain a large table seam, bin shadow, or
        part of the tool as well as the expected screw.  Choosing one connected
        component by an area-weighted score can prefer that contaminant.  Fit
        every nearby component against the expected M4/M8 metric contract and
        then use the nearest valid fit.
        """
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        maximum_distance = max(12.0, 0.35 * min(mask.shape[:2]))
        candidates = []
        rejected = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            distance = float(np.linalg.norm(centers[index] - target))
            if distance > maximum_distance:
                continue
            ys, xs = np.nonzero(labels == index)
            points = np.stack([xs, ys], axis=1).astype(np.float64)
            try:
                fit = self._fit_screw_shape(
                    points,
                    offset,
                    image_shape,
                    expected_class=expected_class,
                )
            except PerceptionError as exc:
                rejected.append(f"area={area}, distance={distance:.1f}: {exc}")
                continue
            candidates.append((distance, -area, fit))

        if not candidates:
            if not rejected:
                raise PerceptionError(
                    f"No pixel-supported {expected_class} near its previous target"
                )
            detail = "; ".join(rejected[:3])
            raise PerceptionError(
                f"No plausible {expected_class} near its previous target; {detail}"
            )
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _candidate_crop(image: np.ndarray, item: dict, padding: float = 2.1):
        height, width = image.shape[:2]
        center = np.asarray(item["center"], float) * [width, height]
        bbox = item.get("bbox")
        if bbox:
            x0, y0, x1, y1 = np.asarray(bbox, float) * [width, height, width, height]
            half_width = max(16.0, abs(x1 - x0) * padding / 2)
            half_height = max(16.0, abs(y1 - y0) * padding / 2)
        else:
            half_width = half_height = 35.0
        left = max(0, int(center[0] - half_width))
        right = min(width, int(center[0] + half_width) + 1)
        top = max(0, int(center[1] - half_height))
        bottom = min(height, int(center[1] + half_height) + 1)
        return image[top:bottom, left:right], np.asarray([left, top]), center - [left, top]

    @staticmethod
    def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
        if color == "red":
            low = cv2.inRange(hsv, (0, 70, 40), (13, 255, 255))
            high = cv2.inRange(hsv, (165, 70, 35), (179, 255, 255))
            return cv2.bitwise_or(low, high)
        if color == "blue":
            return cv2.inRange(hsv, (90, 55, 35), (135, 255, 255))
        if color == "yellow":
            return cv2.inRange(hsv, (15, 60, 50), (48, 255, 255))
        if color == "green":
            return cv2.inRange(hsv, (38, 50, 35), (92, 255, 255))
        raise ValueError(color)

    def _pixel_cube_proposals(self, image_rgb: np.ndarray) -> list[dict]:
        """Find every task-colored 10 mm cube independently of Cosmos counts.

        Cosmos is still required for the scene and screw proposals.  The real
        camera's current crop puts one red cube at the lower crop boundary, so
        using only Cosmos's cropped response can omit that otherwise clear
        cube.  Small red/blue components are unambiguous after excluding the
        much larger bins and restricting detection to the calibrated object
        workspace (with a small edge allowance).
        """
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, (left, top, right, bottom) = self._object_crop(image)
        left = max(0, left - 35)
        right = min(width, right + 35)
        top = max(0, top - 35)
        bottom = min(height, bottom + 65)
        kernel = np.ones((3, 3), np.uint8)
        proposals = []
        for color in ("red", "blue"):
            mask = self._color_mask(hsv, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            candidates = []
            for index in range(1, count):
                x, y, box_width, box_height, area = [int(v) for v in stats[index]]
                center_x, center_y = centers[index]
                ratio = box_width / max(1, box_height)
                if not (180 <= area <= 900):
                    continue
                if not (10 <= box_width <= 42 and 10 <= box_height <= 42):
                    continue
                if not (0.50 <= ratio <= 2.0):
                    continue
                if not (left <= center_x <= right and top <= center_y <= bottom):
                    continue
                side_px = float(np.sqrt(area))
                side_x_mm = self._metric_segment_mm(
                    centers[index], np.asarray([1.0, 0.0]), side_px, image.shape, 0.010
                )
                side_y_mm = self._metric_segment_mm(
                    centers[index], np.asarray([0.0, 1.0]), side_px, image.shape, 0.010
                )
                cube_side_mm = 0.5 * (side_x_mm + side_y_mm)
                if not CUBE_SIDE_RANGE_MM[0] <= cube_side_mm <= CUBE_SIDE_RANGE_MM[1]:
                    continue
                candidates.append((area, x, y, box_width, box_height, center_x, center_y))
            for _, x, y, box_width, box_height, center_x, center_y in candidates:
                proposals.append(
                    {
                        "label": f"{color} cube (workspace HSV proposal)",
                        "sort_class": f"{color}_cube",
                        "center": [center_x / width, center_y / height],
                        "grasp_center": [center_x / width, center_y / height],
                        "bbox": [
                            x / width,
                            y / height,
                            (x + box_width) / width,
                            (y + box_height) / height,
                        ],
                        "graspable": True,
                    }
                )
        return proposals

    def _pixel_screw_proposals(self, image_rgb: np.ndarray) -> list[dict]:
        """Find screw-shaped components across the calibrated source field.

        This is deliberately independent of the requested inventory and of
        Cosmos recall. It is the variable-count recovery path: Cosmos grounds
        the task, while calibrated local geometry decides which visible
        components are plausible screws and the metric fitter decides M4/M8.
        """
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        _, (left, top, right, bottom) = self._object_crop(image)
        # Retain the bounded edge allowance proven by the Policy 6 physical
        # bottom-edge regression frame.
        left = max(0, left - 35)
        right = min(width, right + 35)
        top = max(0, top - 35)
        bottom = min(height, bottom + 65)
        crop = image[top:bottom, left:right]
        light = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        background = float(np.percentile(light, 70.0))
        mask = (light < background - 30.0).astype(np.uint8) * 255
        scale = min(
            width / self.cameras["table"]["size"][0],
            height / self.cameras["table"]["size"][1],
        )
        kernel_size = max(3, int(round(3.0 * scale)) | 1)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), np.uint8),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        proposals = []
        area_scale = max(scale * scale, 1e-6)
        for index in range(1, count):
            x, y, box_width, box_height, area = [int(value) for value in stats[index]]
            if not 45.0 * area_scale <= area <= 1800.0 * area_scale:
                continue
            ys, xs = np.nonzero(labels == index)
            points = np.column_stack((xs, ys)).astype(np.float64)
            center_local = points.mean(axis=0)
            _, singular, vectors = np.linalg.svd(
                points - center_local, full_matrices=False
            )
            axis = vectors[0]
            projection = (points - center_local) @ axis
            span = float(np.ptp(projection))
            elongation = float(singular[0] / max(singular[1], 1e-9))
            if not 10.0 * scale <= span <= 100.0 * scale:
                continue
            if not 1.25 <= elongation <= 8.5:
                continue
            center = center_local + [left, top]
            endpoints = np.stack(
                [
                    center_local + axis * float(projection.min()) + [left, top],
                    center_local + axis * float(projection.max()) + [left, top],
                ]
            )
            proposals.append(
                {
                    "label": "local screw candidate",
                    # Placeholder only. _fit_screw_shape performs the actual
                    # calibrated M4/M8 classification.
                    "sort_class": "m4_screw",
                    "center": (center / [width, height]).tolist(),
                    "grasp_center": (center / [width, height]).tolist(),
                    "bbox": [
                        (left + x) / width,
                        (top + y) / height,
                        (left + x + box_width) / width,
                        (top + y + box_height) / height,
                    ],
                    "axis_endpoints": (endpoints / [width, height]).tolist(),
                    "graspable": True,
                    "source": "calibrated full-field screw scan",
                }
            )
        return proposals

    @staticmethod
    def _component_centers(mask: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Return bounded dark-component seeds without assuming a count."""
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        result = []
        for index in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[index]]
            if not (35 <= area <= 1200):
                continue
            if not (4 <= width <= 80 and 4 <= height <= 80):
                continue
            if (
                x <= 1
                or y <= 1
                or x + width >= mask.shape[1] - 1
                or y + height >= mask.shape[0] - 1
            ):
                continue
            result.append((np.asarray(centers[index], dtype=float), area))
        return result

    def _fit_local_screw_seed(self, bgr: np.ndarray, seed: np.ndarray) -> dict:
        """Fit a seed twice and require stable size, centre, and screw axis."""
        height, width = bgr.shape[:2]
        native_width, native_height = self.cameras["table"]["size"]
        scale = max(0.5, min(width / native_width, height / native_height))
        half = int(round(24.0 * scale))
        left = max(0, int(round(seed[0])) - half)
        right = min(width, int(round(seed[0])) + half + 1)
        top = max(0, int(round(seed[1])) - half)
        bottom = min(height, int(round(seed[1])) + half + 1)
        patch = bgr[top:bottom, left:right]
        if min(patch.shape[:2]) < 20:
            raise PerceptionError("Local screw seed is too close to the image edge")
        light = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[:, :, 0]
        border = np.concatenate((light[0], light[-1], light[:, 0], light[:, -1]))
        background = float(np.median(border))
        target = seed - [left, top]
        kernel = np.ones((3, 3), np.uint8)
        fits = []
        for delta in (15.0, 18.0):
            mask = (light < min(195.0, background - delta)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            selected = self._component_near(mask, target, min_area=20)
            if selected is None:
                raise PerceptionError("Local screw seed has no stable dark silhouette")
            component, area = selected
            ys, xs = np.nonzero(component)
            points = np.stack([xs, ys], axis=1).astype(np.float64)
            fit = self._fit_screw_shape(
                points, np.asarray([left, top], dtype=float), bgr.shape
            )
            fit["silhouette_area_px"] = int(area)
            fits.append(fit)

        first, second = fits
        first_center = np.asarray(first["center"]) * [width, height]
        second_center = np.asarray(second["center"]) * [width, height]
        if first["sort_class"] != second["sort_class"]:
            raise PerceptionError("Local screw class is unstable across thresholds")
        if np.linalg.norm(first_center - second_center) > 5.0 * scale:
            raise PerceptionError("Local screw centre is unstable across thresholds")

        def fitted_yaw(fit):
            points = np.asarray(fit["axis_endpoints"], dtype=float) * [width, height]
            delta = points[1] - points[0]
            return float(np.degrees(np.arctan2(delta[1], delta[0])))

        if self._axis_angle_error(fitted_yaw(first), fitted_yaw(second)) > 12.0:
            raise PerceptionError("Local screw axis is unstable across thresholds")
        if abs(first["length_mm"] - second["length_mm"]) > 6.0:
            raise PerceptionError("Local screw length is unstable across thresholds")
        if np.linalg.norm(second_center - seed) > 10.0 * scale:
            raise PerceptionError("Local screw fit moved too far from its scan seed")

        native = second_center / [width, height] * [native_width, native_height]
        base = pixel_to_base(self.cameras["table"], native, z=SURFACE_Z + 0.006)
        if not (
            0.27 <= base[0] <= 0.52
            and -0.31 <= base[1] <= 0.31
            and np.hypot(base[0], base[1]) <= 0.55
        ):
            raise PerceptionError("Local screw fit is outside the guarded workspace")
        return second

    def _full_frame_local_screws(self, image_rgb: np.ndarray) -> list[dict]:
        """Find strongly supported dark screws over the expanded source field."""
        image = np.asarray(image_rgb, np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        _, (left, top, right, bottom) = self._object_crop(image)
        left = max(0, left - 35)
        right = min(width, right + 35)
        top = max(0, top - 35)
        bottom = min(height, bottom + 65)
        roi = bgr[top:bottom, left:right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        colored = np.zeros(roi.shape[:2], np.uint8)
        for color in ("red", "blue", "yellow", "green"):
            colored = cv2.bitwise_or(colored, self._color_mask(hsv, color))
        colored = cv2.dilate(colored, np.ones((7, 7), np.uint8), iterations=1)
        colored[hsv[:, :, 2] < 80] = 0

        kernel = np.ones((3, 3), np.uint8)
        dark = ((hsv[:, :, 2] < 90) & (colored == 0)).astype(np.uint8) * 255
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            8,
        )
        adaptive[colored > 0] = 0
        for mask in (dark, adaptive):
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, dst=mask)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)

        dark_seeds = self._component_centers(dark)
        adaptive_seeds = self._component_centers(adaptive)
        scale = max(
            0.5,
            min(
                width / self.cameras["table"]["size"][0],
                height / self.cameras["table"]["size"][1],
            ),
        )
        support_radius = 10.0 * scale
        seeds = []
        for dark_center, dark_area in dark_seeds:
            matches = [
                (float(np.linalg.norm(dark_center - center)), center, area)
                for center, area in adaptive_seeds
                if np.linalg.norm(dark_center - center) <= support_radius
            ]
            if not matches:
                continue
            _, adaptive_center, adaptive_area = min(matches, key=lambda item: item[0])
            center = 0.5 * (dark_center + adaptive_center) + [left, top]
            if any(
                np.linalg.norm(center - old[0]) <= LOCAL_SCAN_MATCH_PX * scale
                for old in seeds
            ):
                continue
            seeds.append((center, dark_area, adaptive_area))
        if len(seeds) > MAX_LOCAL_SCREW_CANDIDATES:
            raise PerceptionError(
                "Full-workspace screw scan is ambiguous: "
                f"{len(seeds)} mutually supported dark components"
            )

        candidates = []
        for seed, dark_area, adaptive_area in seeds:
            try:
                fit = self._fit_local_screw_seed(bgr, seed)
            except PerceptionError:
                continue
            center = np.asarray(fit["center"]) * [width, height]
            if any(
                np.linalg.norm(center - np.asarray(item["_pixel_center"]))
                <= LOCAL_SCAN_MATCH_PX * scale
                for item in candidates
            ):
                continue
            candidates.append(
                {
                    "label": f"{fit['sort_class']} (stable full-field scan)",
                    "sort_class": fit["sort_class"],
                    "center": fit["center"].tolist(),
                    "grasp_center": fit["grasp_center"].tolist(),
                    "axis_endpoints": fit["axis_endpoints"].tolist(),
                    "graspable": True,
                    "source": "calibrated stable full-workspace scan",
                    "visual_size_px": fit["head_width_px"],
                    "visual_span_px": fit["span_px"],
                    "head_width_mm": fit["head_width_mm"],
                    "shank_width_mm": fit["shank_width_mm"],
                    "screw_length_mm": fit["length_mm"],
                    "grasp_feature": fit["grasp_feature"],
                    "local_scan_support": {
                        "dark_area_px": int(dark_area),
                        "adaptive_area_px": int(adaptive_area),
                        "thresholds": [15, 18],
                    },
                    "_pixel_center": center,
                }
            )
        return candidates

    def _metric_segment_mm(
        self,
        center_px: np.ndarray,
        direction_px: np.ndarray,
        length_px: float,
        image_shape,
        height_above_table_m: float,
    ) -> float:
        """Convert a short image segment to calibrated base-frame millimetres."""
        height, width = image_shape[:2]
        native_width, native_height = self.cameras["table"]["size"]
        direction = np.asarray(direction_px, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        endpoints = np.stack(
            [
                np.asarray(center_px, dtype=float) - direction * length_px / 2.0,
                np.asarray(center_px, dtype=float) + direction * length_px / 2.0,
            ]
        )
        native = endpoints / [width, height] * [native_width, native_height]
        base = np.stack(
            [
                pixel_to_base(
                    self.cameras["table"],
                    point,
                    z=SURFACE_Z + height_above_table_m,
                )
                for point in native
            ]
        )
        return float(np.linalg.norm(base[1, :2] - base[0, :2]) * 1000.0)

    @staticmethod
    def _screw_class_from_metric_shape(
        head_width_mm: float,
        shank_width_mm: float,
        length_mm: float,
    ) -> str:
        matches = []
        for sort_class, geometry in SCREW_GEOMETRY_MM.items():
            if (
                geometry["head_range"][0]
                <= head_width_mm
                <= geometry["head_range"][1]
                and geometry["shank_range"][0]
                <= shank_width_mm
                <= geometry["shank_range"][1]
                and geometry["length_range"][0]
                <= length_mm
                <= geometry["length_range"][1]
            ):
                matches.append(sort_class)
        if len(matches) != 1:
            raise PerceptionError(
                "Screw metric dimensions are ambiguous or unsupported: "
                f"head={head_width_mm:.1f} mm, shank={shank_width_mm:.1f} mm, "
                f"length={length_mm:.1f} mm, matches={matches}"
            )
        return matches[0]

    @staticmethod
    def _screw_class_from_observed_silhouette(
        head_width_mm: float,
        shank_width_mm: float,
        length_mm: float,
    ) -> str:
        matches = []
        for sort_class, ranges in OBSERVED_SCREW_SILHOUETTE_MM.items():
            if (
                ranges["head_range"][0]
                <= head_width_mm
                <= ranges["head_range"][1]
                and ranges["shank_range"][0]
                <= shank_width_mm
                <= ranges["shank_range"][1]
                and ranges["length_range"][0]
                <= length_mm
                <= ranges["length_range"][1]
            ):
                matches.append(sort_class)
        if len(matches) != 1:
            raise PerceptionError(
                "Observed screw silhouette is ambiguous or unsupported: "
                f"head={head_width_mm:.1f} mm, shank={shank_width_mm:.1f} mm, "
                f"length={length_mm:.1f} mm, matches={matches}"
            )
        return matches[0]

    def _fit_screw_shape(
        self,
        points: np.ndarray,
        offset: np.ndarray,
        image_shape,
        expected_class: str | None = None,
    ) -> dict:
        """Fit screw axis/dimensions and put the grasp at the shaft midpoint."""
        points = np.asarray(points, dtype=np.float64)
        center_local = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points - center_local, full_matrices=False)
        axis = vectors[0]
        perpendicular_axis = np.asarray([-axis[1], axis[0]])
        projection = (points - center_local) @ axis
        perpendicular = (points - center_local) @ perpendicular_axis
        span = float(np.ptp(projection))
        if span < 8.0:
            raise PerceptionError(f"Screw silhouette is too short: {span:.1f} px")
        low = projection < projection.min() + 0.28 * span
        high = projection > projection.max() - 0.28 * span

        def end_width(selection):
            values = perpendicular[selection]
            return float(np.percentile(values, 95) - np.percentile(values, 5))

        low_width = end_width(low)
        high_width = end_width(high)
        head_at_low = low_width >= high_width
        head_width_px = max(low_width, high_width)
        shank_width_px = min(low_width, high_width)
        full_center = center_local + offset
        endpoints = np.stack(
            [
                center_local + axis * float(projection.min()) + offset,
                center_local + axis * float(projection.max()) + offset,
            ]
        )
        # Use a common 6 mm observation plane for classification.  It is
        # halfway between the supplied 4 mm M4 and 8 mm M8 head heights and
        # changes the local pixel scale by much less than the guarded ranges.
        metric_height_m = 0.006
        head_width_mm = self._metric_segment_mm(
            full_center,
            perpendicular_axis,
            head_width_px,
            image_shape,
            metric_height_m,
        )
        shank_width_mm = self._metric_segment_mm(
            full_center,
            perpendicular_axis,
            shank_width_px,
            image_shape,
            metric_height_m,
        )
        length_mm = self._metric_segment_mm(
            full_center, axis, span, image_shape, metric_height_m
        )
        try:
            measured_class = self._screw_class_from_metric_shape(
                head_width_mm, shank_width_mm, length_mm
            )
        except PerceptionError:
            measured_class = self._screw_class_from_observed_silhouette(
                head_width_mm, shank_width_mm, length_mm
            )
        if expected_class is not None and measured_class != expected_class:
            raise PerceptionError(
                f"Expected {expected_class}, but metric silhouette is {measured_class}"
            )
        sort_class = expected_class or measured_class
        geometry = SCREW_GEOMETRY_MM[sort_class]
        fraction_from_head = (
            geometry["head_height"] + geometry["shank_length"] / 2.0
        ) / geometry["total_length"]
        if head_at_low:
            shaft_projection = projection.min() + fraction_from_head * span
        else:
            shaft_projection = projection.max() - fraction_from_head * span

        # Centre laterally on the shaft using only pixels around the desired
        # axial position.  This avoids the wide head biasing the grasp point.
        band = np.abs(projection - shaft_projection) <= 0.12 * span
        if int(np.count_nonzero(band)) < 6:
            raise PerceptionError("Too few shaft pixels for a centred grasp")
        shaft_perpendicular = perpendicular[band]
        lateral_center = 0.5 * (
            np.percentile(shaft_perpendicular, 5)
            + np.percentile(shaft_perpendicular, 95)
        )
        shaft_center = (
            center_local
            + axis * float(shaft_projection)
            + perpendicular_axis * float(lateral_center)
            + offset
        )
        height, width = image_shape[:2]
        return {
            "sort_class": sort_class,
            "center": full_center / [width, height],
            "grasp_center": shaft_center / [width, height],
            "axis_endpoints": endpoints / [width, height],
            "head_width_px": head_width_px,
            "shank_width_px": shank_width_px,
            "span_px": span,
            "head_width_mm": head_width_mm,
            "shank_width_mm": shank_width_mm,
            "length_mm": length_mm,
            "grasp_feature": "shaft_midpoint",
        }

    def _refine_objects(
        self,
        image_rgb: np.ndarray,
        proposals: list[dict],
        *,
        allow_empty: bool = False,
    ) -> list[dict]:
        """Fuse Cosmos and full-field proposals, then classify metric geometry."""
        bgr = cv2.cvtColor(np.asarray(image_rgb, np.uint8), cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        kernel = np.ones((3, 3), np.uint8)
        # The dual-threshold scan is the high-confidence inventory.  Build the
        # broader/Cosmos inventory independently so a robot-edge or shadow
        # component cannot be added merely because its centre is new.  We use
        # the broader inventory only when it actually recovers more verified
        # screw silhouettes than the strict scan.
        stable_refined = self._full_frame_local_screws(image_rgb)
        fallback_refined = []
        local_proposals = self._pixel_screw_proposals(image_rgb)
        native_width, native_height = self.cameras["table"]["size"]
        duplicate_radius_px = 18.0 * min(
            width / native_width, height / native_height
        )
        screw_proposals = []
        proposal_duplicate_px = 12.0 * min(
            width / native_width, height / native_height
        )
        for item in local_proposals + list(proposals):
            if not str(item.get("sort_class", "")).endswith("_screw"):
                continue
            center = np.asarray(item["center"], dtype=float) * [width, height]
            if any(
                np.linalg.norm(
                    center
                    - np.asarray(existing["center"], dtype=float) * [width, height]
                )
                <= proposal_duplicate_px
                for existing in screw_proposals
            ):
                continue
            screw_proposals.append(item)
        proposals = screw_proposals
        for proposal in proposals:
            patch, offset, target = self._candidate_crop(bgr, proposal)
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            color_hits = []
            if str(proposal.get("sort_class", "")).endswith("_cube"):
                for color in ("red", "blue"):
                    mask = self._color_mask(hsv, color)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    selected = self._component_near(mask, target, min_area=18)
                    if selected is not None:
                        component, area = selected
                        ys, xs = np.nonzero(component)
                        center = np.asarray([xs.mean(), ys.mean()])
                        color_hits.append(
                            (
                                float(np.linalg.norm(center - target)),
                                -area,
                                color,
                                component,
                                area,
                            )
                        )
            if color_hits:
                _, _, color, component, area = min(color_hits)
                ys, xs = np.nonzero(component)
                points = np.stack([xs, ys], axis=1).astype(np.float32)
                center = points.mean(axis=0) + offset
                full_center = [center[0] / width, center[1] / height]
                side_px = float(np.sqrt(area))
                side_x_mm = self._metric_segment_mm(
                    center, np.asarray([1.0, 0.0]), side_px, bgr.shape, 0.010
                )
                side_y_mm = self._metric_segment_mm(
                    center, np.asarray([0.0, 1.0]), side_px, bgr.shape, 0.010
                )
                cube_side_mm = 0.5 * (side_x_mm + side_y_mm)
                if not CUBE_SIDE_RANGE_MM[0] <= cube_side_mm <= CUBE_SIDE_RANGE_MM[1]:
                    continue
                candidate = {
                    "label": f"{color} cube (Cosmos + pixel verified)",
                    "sort_class": f"{color}_cube",
                    "center": full_center,
                    "grasp_center": full_center,
                    "bbox": proposal.get("bbox"),
                    "axis_endpoints": None,
                    "graspable": True,
                    "source": "Cosmos3-Nano + HSV verification",
                    "visual_size_px": float(np.sqrt(area)),
                    "visual_span_px": float(
                        max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
                    ),
                    "cube_side_mm": cube_side_mm,
                    "grasp_feature": "cube_center",
                    "proposal_grasp_center": proposal.get(
                        "grasp_center", proposal.get("center")
                    ),
                    "proposal_center": proposal.get("center"),
                    "cosmos_axis_endpoints": proposal.get("axis_endpoints"),
                }
                feature = ("cube", color, center)
            else:
                lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
                light = lab[:, :, 0]
                border = np.concatenate((light[0], light[-1], light[:, 0], light[:, -1]))
                background = float(np.median(border))
                mask = (light < min(195.0, background - 18.0)).astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                selected = self._component_near(mask, target, min_area=12)
                if selected is None:
                    continue
                component, area = selected
                ys, xs = np.nonzero(component)
                points = np.stack([xs, ys], axis=1).astype(np.float64)
                try:
                    fit = self._fit_screw_shape(points, offset, bgr.shape)
                except PerceptionError:
                    continue
                center = fit["center"] * [width, height]
                candidate = {
                    "label": f"{fit['sort_class']} (metric-size, shaft grasp)",
                    "sort_class": fit["sort_class"],
                    "center": fit["center"].tolist(),
                    "grasp_center": fit["grasp_center"].tolist(),
                    "bbox": proposal.get("bbox"),
                    "axis_endpoints": fit["axis_endpoints"].tolist(),
                    "graspable": True,
                    "source": "Cosmos3-Nano + calibrated metric silhouette",
                    "silhouette_area_px": int(area),
                    "visual_size_px": fit["head_width_px"],
                    "visual_span_px": fit["span_px"],
                    "head_width_mm": fit["head_width_mm"],
                    "shank_width_mm": fit["shank_width_mm"],
                    "screw_length_mm": fit["length_mm"],
                    "grasp_feature": fit["grasp_feature"],
                    "proposal_grasp_center": proposal.get(
                        "grasp_center", proposal.get("center")
                    ),
                    "proposal_center": proposal.get("center"),
                    "cosmos_axis_endpoints": proposal.get("axis_endpoints"),
                }
                feature = ("screw", None, center)

            if any(
                np.linalg.norm(feature[2] - entry["_pixel_center"]) < 12.0
                for entry in fallback_refined
            ):
                continue
            candidate["_pixel_center"] = feature[2]
            fallback_refined.append(candidate)

        if len(stable_refined) >= len(fallback_refined):
            refined = stable_refined
        else:
            refined = list(stable_refined)
            for candidate in fallback_refined:
                if any(
                    np.linalg.norm(
                        candidate["_pixel_center"] - entry["_pixel_center"]
                    ) < 12.0
                    for entry in refined
                ):
                    continue
                refined.append(candidate)

        screws = [item for item in refined if item["sort_class"] in SCREW_GEOMETRY_MM]
        counts = Counter(item["sort_class"] for item in refined)
        if not refined:
            if allow_empty:
                return []
            raise PerceptionError("No task-relevant objects were verified")
        if len(refined) > MAX_TASK_OBJECTS or any(
            count > MAX_OBJECTS_PER_CLASS for count in counts.values()
        ):
            raise PerceptionError(
                f"Dynamic inventory exceeds guarded limits: {dict(counts)}"
            )
        for item in refined:
            item.pop("_pixel_center", None)
            item.pop("silhouette_area_px", None)
        return refined

    def _detect_bins(self, image_rgb: np.ndarray) -> list[dict]:
        """Find the red and green destination bins; reject ambiguity."""
        image = np.asarray(image_rgb, np.uint8)
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        kernel = np.ones((3, 3), np.uint8)
        _, (workspace_left, workspace_top, workspace_right, workspace_bottom) = (
            self._object_crop(image)
        )
        bins = []
        for color in dict.fromkeys(self.active_class_to_bin.values()):
            mask = self._color_mask(hsv, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            candidates = []
            for index in range(1, count):
                x, y, box_width, box_height, area = [int(v) for v in stats[index]]
                center_y = y + box_height / 2
                center_x = x + box_width / 2
                ratio = box_width / max(1, box_height)
                if not (600 <= area <= 25000):
                    continue
                if box_width < 20 or box_height < 20 or not (0.4 <= ratio <= 2.5):
                    continue
                if not (
                    workspace_left - 40 <= center_x <= workspace_right + 40
                    and workspace_top - 100 <= center_y <= workspace_bottom
                ):
                    continue
                candidates.append((area, x, y, box_width, box_height))
            if not candidates:
                raise PerceptionError(f"No pixel-verified {color} bin")
            area, x, y, box_width, box_height = max(candidates)
            center = [(x + box_width / 2) / width, (y + box_height / 2) / height]
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
                    "label": f"{color} bin (HSV verified)",
                    "color": color,
                    "center": center,
                    "bbox": [x / width, y / height, (x + box_width) / width, (y + box_height) / height],
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
    def _exclude_bin_contents(
        objects: list[dict], bins: list[dict], image_shape: tuple[int, ...]
    ) -> tuple[list[dict], list[dict]]:
        """Remove detections inside either destination bin before recovery.

        The final audit searches a wider field than the initial Cosmos crop.
        After a successful drop it can therefore see a screw inside a bin and
        misinterpret that already-sorted object as a residual source object.
        Use the independently verified red/green bin silhouettes as a hard
        no-pick zone, expanded slightly for the open interior and shadows.
        """
        height, width = image_shape[:2]
        no_pick_boxes = []
        for item in bins:
            bbox = np.asarray(item.get("bbox", []), dtype=float)
            if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
                raise PerceptionError(f"Missing verified bin bbox: {item}")
            x0, y0, x1, y1 = bbox * [width, height, width, height]
            if x1 <= x0 or y1 <= y0:
                raise PerceptionError(f"Invalid verified bin bbox: {item}")
            margin_x = max(12.0, 0.20 * (x1 - x0))
            margin_y = max(12.0, 0.20 * (y1 - y0))
            no_pick_boxes.append(
                (x0 - margin_x, y0 - margin_y, x1 + margin_x, y1 + margin_y)
            )

        kept, excluded = [], []
        for item in objects:
            point = np.asarray(
                item.get("grasp_center", item.get("center", [])), dtype=float
            )
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise PerceptionError(f"Missing object centre during bin exclusion: {item}")
            x, y = point * [width, height]
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in no_pick_boxes):
                excluded.append(item)
            else:
                kept.append(item)
        return kept, excluded

    def _cosmos_infer(self, image_rgb: np.ndarray, task: Sort4Task) -> dict:
        """Original Cosmos proposal path, retained as a guarded fallback."""
        crop_rgb, crop_box = self._object_crop(image_rgb)
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise PerceptionError("Could not encode the table image")
        payload = {
            "instruction": task.instruction,
            "image_jpeg_base64": base64.b64encode(encoded).decode("ascii"),
        }
        try:
            response = requests.post(
                self.url + "/infer-sort4", json=payload, timeout=180
            )
            result = response.json()
            response.raise_for_status()
        except (requests.RequestException, ValueError) as exc:
            detail = result.get("error") if "result" in locals() else str(exc)
            raise PerceptionError(f"Cosmos sort4 service failed: {detail}") from exc
        if result.get("status") != "PASS":
            raise PerceptionError(result.get("error", "Cosmos rejected the scene"))
        mapped = self._map_crop_scene(result, crop_box, np.asarray(image_rgb).shape)
        allowed = set(task.class_to_bin)
        mapped["ignored_objects"] = [
            {
                "label": str(item.get("label", "distractor")),
                "sort_class": str(item.get("sort_class", "ignore")),
                "center": item.get("center"),
            }
            for item in mapped.get("objects", [])
            if item.get("sort_class") not in allowed
        ]
        relevant = [
            item for item in mapped.get("objects", [])
            if item.get("sort_class") in allowed
        ]
        mapped["objects"] = self._refine_objects(image_rgb, relevant)
        mapped["bins"] = self._detect_bins(image_rgb)
        mapped["object_crop_pixels"] = list(crop_box)
        mapped["verification"] = "Cosmos proposals + pixel evidence"
        return mapped

    def infer(self, image_rgb: np.ndarray, task: Sort4Task) -> dict:
        """Prefer the proven full-field metric screw scan over Cosmos latency.

        Policy 8 already has a stricter local detector than the new crop
        classifier: it requires two independent pixel masks and calibrated
        M4/M8 geometry.  Use it first and invoke the original Cosmos path only
        when no safe local inventory can be formed.
        """
        self.active_class_to_bin = dict(task.class_to_bin)
        self.active_bin_profiles = {}
        try:
            objects = self._refine_objects(image_rgb, [], allow_empty=False)
            bins = self._detect_bins(image_rgb)
            objects, excluded = self._exclude_bin_contents(
                objects, bins, np.asarray(image_rgb).shape
            )
            if not objects:
                raise PerceptionError("Local scan found only objects inside bin no-pick zones")
        except PerceptionError as local_error:
            result = self._cosmos_infer(image_rgb, task)
            result["local_primary_error"] = str(local_error)
            result["verification"] = (
                "local metric scan failed; Cosmos proposals + pixel evidence fallback"
            )
            return result

        return {
            "status": "PASS",
            "objects": objects,
            "bins": bins,
            "ignored_objects": [
                {
                    "label": str(item.get("label", "object inside destination bin")),
                    "sort_class": str(item.get("sort_class", "ignore")),
                    "center": item.get("center"),
                }
                for item in excluded
            ],
            "verification": (
                "local dual-mask full-field scan + calibrated metric geometry; "
                "Cosmos not required"
            ),
        }

    def discover_local_targets(
        self, image_rgb: np.ndarray, task: Sort4Task
    ) -> list[SortTarget]:
        """Inventory every currently visible screw without a fixed count."""
        self.active_class_to_bin = dict(task.class_to_bin)
        objects = self._refine_objects(image_rgb, [], allow_empty=True)
        bins = self._detect_bins(image_rgb)
        objects, _ = self._exclude_bin_contents(
            objects, bins, np.asarray(image_rgb).shape
        )
        if not objects:
            return []
        scene = {
            "objects": objects,
            "bins": bins,
            "verification": "calibrated full-field screw scan outside bin no-pick zones",
        }
        return self.to_targets(scene, task)

    @staticmethod
    def _point(item: dict, key: str) -> np.ndarray:
        point = np.asarray(item.get(key, []), dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise PerceptionError(f"Missing {key} in detection: {item}")
        if np.any(point < 0.0) or np.any(point > 1.0):
            raise PerceptionError(f"Detection {key} is not normalized: {point}")
        return point

    @staticmethod
    def _canonical_axis_yaw(yaw: float) -> float:
        """Parallel jaws are unchanged when the fitted axis is reversed."""
        return float((yaw + 90.0) % 180.0 - 90.0)

    @staticmethod
    def _axis_angle_error(a: float, b: float) -> float:
        return float(abs((a - b + 90.0) % 180.0 - 90.0))

    @staticmethod
    def _validate_separation(targets: list[SortTarget], minimum_mm: float = 18.0):
        for left_index, left in enumerate(targets):
            for right in targets[left_index + 1 :]:
                separation = float(
                    np.linalg.norm(
                        np.asarray(left.object_xy_mm)
                        - np.asarray(right.object_xy_mm)
                    )
                )
                if separation < minimum_mm:
                    raise PerceptionError(
                        f"Object targets overlap or are too close: {separation:.1f} mm"
                    )

    def _refine_remaining_target(
        self, image_rgb: np.ndarray, target: SortTarget
    ) -> dict:
        """Fit one expected class near its previous calibrated position."""
        image = np.asarray(image_rgb, np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise PerceptionError(f"Invalid table RGB image shape: {image.shape}")
        height, width = image.shape[:2]
        cam = self.cameras["table"]
        native_width, native_height = cam["size"]
        image_scale = max(
            1.0, min(width / native_width, height / native_height)
        )
        object_height = OBJECT_HEIGHT_M[target.sort_class]
        predicted_native = project(
            cam,
            [[
                target.object_xy_mm[0] / 1000.0,
                target.object_xy_mm[1] / 1000.0,
                SURFACE_Z + object_height,
            ]],
        )[0]
        predicted = predicted_native / [native_width, native_height] * [width, height]
        if (
            predicted[0] < 35
            or predicted[0] >= width - 35
            or predicted[1] < 35
            or predicted[1] >= height - 35
        ):
            raise PerceptionError(
                f"Projected {target.label} is outside the usable image: {predicted}"
            )
        half = 32.0 * image_scale
        proposal = {
            "center": (predicted / [width, height]).tolist(),
            "bbox": [
                (predicted[0] - half) / width,
                (predicted[1] - half) / height,
                (predicted[0] + half) / width,
                (predicted[1] + half) / height,
            ],
        }
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        patch, offset, local_target = self._candidate_crop(bgr, proposal)
        if patch.size == 0:
            raise PerceptionError(f"Empty relocalization crop for {target.label}")
        kernel = np.ones((3, 3), np.uint8)

        if target.sort_class.endswith("_cube"):
            color = target.sort_class.split("_", 1)[0]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mask = self._color_mask(hsv, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            selected = self._component_near(mask, local_target, min_area=18)
            if selected is None:
                raise PerceptionError(
                    f"No pixel-supported {target.sort_class} near its previous target"
                )
            component, area = selected
            ys, xs = np.nonzero(component)
            points = np.stack([xs, ys], axis=1).astype(np.float64)
            center = points.mean(axis=0) + offset
            span = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
            if span > 80.0 * image_scale:
                raise PerceptionError(
                    f"{target.sort_class} silhouette is merged or oversized: {span:.1f} px"
                )
            side_px = float(np.sqrt(area))
            side_x_mm = self._metric_segment_mm(
                center, np.asarray([1.0, 0.0]), side_px, image.shape, 0.010
            )
            side_y_mm = self._metric_segment_mm(
                center, np.asarray([0.0, 1.0]), side_px, image.shape, 0.010
            )
            cube_side_mm = 0.5 * (side_x_mm + side_y_mm)
            if not CUBE_SIDE_RANGE_MM[0] <= cube_side_mm <= CUBE_SIDE_RANGE_MM[1]:
                raise PerceptionError(
                    f"{target.sort_class} is not a supported 10 mm cube: "
                    f"{cube_side_mm:.1f} mm"
                )
            return {
                "grasp_center": center / [width, height],
                "axis_endpoints": None,
                "visual_size_px": float(np.sqrt(area)),
                "visual_span_px": span,
                "cube_side_mm": cube_side_mm,
                "grasp_feature": "cube_center",
            }

        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0]
        border = np.concatenate((light[0], light[-1], light[:, 0], light[:, -1]))
        background = float(np.median(border))
        mask = (light < min(195.0, background - 18.0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        fit = self._fit_expected_screw_component_near(
            mask,
            local_target,
            offset,
            image.shape,
            target.sort_class,
        )
        span = float(fit["span_px"])
        if not 8.0 * image_scale <= span <= 80.0 * image_scale:
            raise PerceptionError(
                f"{target.sort_class} silhouette span is unsafe: {span:.1f} px"
            )
        return {
            "grasp_center": fit["grasp_center"],
            "axis_endpoints": fit["axis_endpoints"],
            "visual_size_px": fit["head_width_px"],
            "visual_span_px": span,
            "head_width_mm": fit["head_width_mm"],
            "shank_width_mm": fit["shank_width_mm"],
            "screw_length_mm": fit["length_mm"],
            "grasp_feature": fit["grasp_feature"],
        }

    def relocalize_targets(
        self, image_rgb: np.ndarray, targets: list[SortTarget]
    ) -> list[SortTarget]:
        """Refresh all remaining objects and bins without another Cosmos call."""
        if not targets:
            return []
        image = np.asarray(image_rgb, np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise PerceptionError(f"Invalid table RGB image shape: {image.shape}")
        cam = self.cameras["table"]
        native_width, native_height = cam["size"]

        def base(point, height_above_table_m):
            return pixel_to_base(
                cam,
                [point[0] * native_width, point[1] * native_height],
                z=SURFACE_Z + height_above_table_m,
            )

        bins = {item["color"]: item for item in self._detect_bins(image)}
        full_frame_fallback = None
        used_fallback_indices = set()

        def refine_with_fallback(old: SortTarget) -> dict:
            """Use a stable full-frame fit when a local crop merges with a seam."""
            nonlocal full_frame_fallback
            try:
                return self._refine_remaining_target(image, old)
            except PerceptionError as local_error:
                if not str(local_error).startswith("No plausible"):
                    raise
                if full_frame_fallback is None:
                    scanned = self._full_frame_local_screws(image)
                    full_frame_fallback, _ = self._exclude_bin_contents(
                        scanned, list(bins.values()), image.shape
                    )

                compatible = []
                object_height = OBJECT_HEIGHT_M[old.sort_class]
                for index, item in enumerate(full_frame_fallback):
                    if index in used_fallback_indices:
                        continue
                    if item.get("sort_class") != old.sort_class:
                        continue
                    candidate_xyz = base(item["grasp_center"], object_height)
                    shift_mm = float(
                        np.linalg.norm(
                            candidate_xyz[:2] * 1000.0
                            - np.asarray(old.object_xy_mm)
                        )
                    )
                    if shift_mm <= 12.0:
                        compatible.append((shift_mm, index, item))
                if not compatible:
                    raise local_error

                _, index, item = min(compatible, key=lambda entry: entry[0])
                used_fallback_indices.add(index)
                return {
                    "grasp_center": np.asarray(item["grasp_center"], dtype=float),
                    "axis_endpoints": np.asarray(
                        item["axis_endpoints"], dtype=float
                    ),
                    "visual_size_px": float(item["visual_size_px"]),
                    "visual_span_px": float(item["visual_span_px"]),
                    "head_width_mm": float(item["head_width_mm"]),
                    "shank_width_mm": float(item["shank_width_mm"]),
                    "screw_length_mm": float(item["screw_length_mm"]),
                    "grasp_feature": str(item["grasp_feature"]),
                }

        refreshed = []
        for old in targets:
            fit = refine_with_fallback(old)
            object_height = OBJECT_HEIGHT_M[old.sort_class]
            object_xyz = base(fit["grasp_center"], object_height)
            new_xy_mm = np.asarray(object_xyz[:2]) * 1000.0
            shift_mm = float(
                np.linalg.norm(new_xy_mm - np.asarray(old.object_xy_mm))
            )
            if shift_mm > 12.0:
                raise PerceptionError(
                    f"{old.label} moved {shift_mm:.1f} mm; refusing stale association"
                )
            if old.visual_size_px > 0.0:
                size_ratio = fit["visual_size_px"] / old.visual_size_px
                if not 0.65 <= size_ratio <= 1.50:
                    raise PerceptionError(
                        f"{old.label} changed apparent size by {size_ratio:.2f}x"
                    )

            yaw = 0.0
            endpoints = fit["axis_endpoints"]
            if endpoints is not None:
                p0 = base(endpoints[0], object_height)
                p1 = base(endpoints[1], object_height)
                delta = p1[:2] - p0[:2]
                yaw = self._canonical_axis_yaw(
                    float(np.degrees(np.arctan2(delta[1], delta[0])))
                )

            bin_item = bins.get(old.bin_color)
            if bin_item is None:
                raise PerceptionError(f"Missing refreshed {old.bin_color} bin")
            if str(bin_item["profile"]) != old.bin_profile:
                raise PerceptionError(
                    f"{old.bin_color} bin profile changed from {old.bin_profile} "
                    f"to {bin_item['profile']}"
                )
            bin_xyz = base(
                self._point(bin_item, "center"),
                float(bin_item["visual_midpoint_m"]),
            )
            new_bin_xy_mm = np.asarray(bin_xyz[:2]) * 1000.0
            bin_shift_mm = float(
                np.linalg.norm(new_bin_xy_mm - np.asarray(old.bin_xy_mm))
            )
            if bin_shift_mm > MAX_BIN_EVIDENCE_SHIFT_MM:
                raise PerceptionError(
                    f"{old.bin_color} bin moved {bin_shift_mm:.1f} mm"
                )

            refreshed.append(
                SortTarget(
                    sort_class=old.sort_class,
                    label=old.label,
                    object_xy_mm=(float(new_xy_mm[0]), float(new_xy_mm[1])),
                    object_yaw_deg=yaw,
                    bin_color=old.bin_color,
                    # A destination bin cannot move during a trial.  Use the
                    # initial, unoccluded centre for motion; the refreshed
                    # component is presence/identity evidence only.
                    bin_xy_mm=old.bin_xy_mm,
                    bin_profile=old.bin_profile,
                    bin_footprint_mm=old.bin_footprint_mm,
                    bin_body_height_mm=old.bin_body_height_mm,
                    bin_release_z_mm=old.bin_release_z_mm,
                    visual_size_px=float(fit["visual_size_px"]),
                    visual_span_px=float(fit["visual_span_px"]),
                    initial_refinement_mm=old.initial_refinement_mm,
                    initial_yaw_refinement_deg=old.initial_yaw_refinement_deg,
                    relocalization_mm=shift_mm,
                    relocalization_yaw_deg=self._axis_angle_error(
                        yaw, old.object_yaw_deg
                    ),
                    bin_relocalization_mm=bin_shift_mm,
                    head_width_mm=float(fit.get("head_width_mm", 0.0)),
                    shank_width_mm=float(fit.get("shank_width_mm", 0.0)),
                    screw_length_mm=float(fit.get("screw_length_mm", 0.0)),
                    cube_side_mm=float(fit.get("cube_side_mm", 0.0)),
                    grasp_feature=str(fit.get("grasp_feature", "center")),
                )
            )
        self._validate_separation(refreshed)
        return refreshed

    def target_is_present(
        self, image_rgb: np.ndarray, target: SortTarget
    ) -> bool:
        """Return whether the expected object still occupies its source pose.

        A clean absence is different from an ambiguous/merged silhouette.  Only
        the explicit no-pixel-evidence condition counts as removed; every other
        perception problem propagates and stops motion.
        """
        try:
            self._refine_remaining_target(image_rgb, target)
        except PerceptionError as exc:
            if str(exc).startswith("No pixel-supported"):
                return False
            raise
        return True

    def residual_targets(
        self, image_rgb: np.ndarray, task: Sort4Task
    ) -> list[SortTarget]:
        """Discover all remaining screws, including initial Cosmos omissions."""
        return self.discover_local_targets(image_rgb, task)

    def to_targets(self, scene: dict, task: Sort4Task) -> list[SortTarget]:
        bins_by_color: dict[str, dict] = {}
        for item in scene.get("bins", []):
            color = str(item.get("color", item.get("colour", ""))).lower()
            if color in set(task.class_to_bin.values()):
                if color in bins_by_color:
                    raise PerceptionError(f"Detected more than one {color} bin")
                bins_by_color[color] = item
        missing_bins = sorted(set(task.class_to_bin.values()) - set(bins_by_color))
        if missing_bins:
            raise PerceptionError("Missing destination bins: " + ", ".join(missing_bins))

        allowed = set(task.class_to_bin)
        objects = [
            item
            for item in scene.get("objects", [])
            if item.get("sort_class") in allowed and item.get("graspable", True)
        ]
        counts = Counter(item["sort_class"] for item in objects)
        if not objects:
            raise PerceptionError("No task-relevant objects detected")
        if len(objects) > MAX_TASK_OBJECTS or any(
            count > MAX_OBJECTS_PER_CLASS for count in counts.values()
        ):
            raise PerceptionError(
                f"Dynamic inventory exceeds guarded limits: {dict(counts)}"
            )

        cam = self.cameras["table"]
        width, height = cam["size"]

        def base(point: np.ndarray, height_above_table_m: float = 0.0) -> np.ndarray:
            return pixel_to_base(
                cam,
                [point[0] * width, point[1] * height],
                z=SURFACE_Z + height_above_table_m,
            )

        targets = []
        class_order = {name: index for index, name in enumerate(task.class_to_bin)}
        for item in objects:
            sort_class = item["sort_class"]
            center_key = "grasp_center" if item.get("grasp_center") else "center"
            object_height = OBJECT_HEIGHT_M[sort_class]
            object_xyz = base(self._point(item, center_key), object_height)
            initial_refinement_mm = 0.0
            proposal_center = item.get("proposal_center")
            if proposal_center is not None:
                refined_center_xyz = base(self._point(item, "center"), object_height)
                proposal_xyz = base(
                    self._point({"value": proposal_center}, "value"), object_height
                )
                initial_refinement_mm = float(
                    np.linalg.norm((refined_center_xyz - proposal_xyz)[:2]) * 1000.0
                )
                if initial_refinement_mm > 12.0:
                    raise PerceptionError(
                        f"Initial {sort_class} refinement moved {initial_refinement_mm:.1f} mm"
                    )
            color = task.class_to_bin[sort_class]
            bin_item = bins_by_color[color]
            # The HSV component's bounding-box centre lies halfway up its
            # detected bin body. Each supported profile supplies that plane.
            bin_xyz = base(
                self._point(bin_item, "center"),
                float(bin_item["visual_midpoint_m"]),
            )
            yaw = 0.0
            endpoints = item.get("axis_endpoints") or []
            if len(endpoints) == 2:
                p0 = base(self._point({"value": endpoints[0]}, "value"), object_height)
                p1 = base(self._point({"value": endpoints[1]}, "value"), object_height)
                delta = p1[:2] - p0[:2]
                yaw = self._canonical_axis_yaw(
                    float(np.degrees(np.arctan2(delta[1], delta[0])))
                )
            initial_yaw_refinement_deg = 0.0
            cosmos_endpoints = item.get("cosmos_axis_endpoints") or []
            if len(cosmos_endpoints) == 2:
                c0 = base(
                    self._point({"value": cosmos_endpoints[0]}, "value"),
                    object_height,
                )
                c1 = base(
                    self._point({"value": cosmos_endpoints[1]}, "value"),
                    object_height,
                )
                cdelta = c1[:2] - c0[:2]
                cosmos_yaw = self._canonical_axis_yaw(
                    float(np.degrees(np.arctan2(cdelta[1], cdelta[0])))
                )
                initial_yaw_refinement_deg = self._axis_angle_error(
                    yaw, cosmos_yaw
                )
            targets.append(
                SortTarget(
                    sort_class=sort_class,
                    label=str(item.get("label", sort_class)),
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
                    initial_refinement_mm=initial_refinement_mm,
                    initial_yaw_refinement_deg=initial_yaw_refinement_deg,
                    head_width_mm=float(item.get("head_width_mm", 0.0)),
                    shank_width_mm=float(item.get("shank_width_mm", 0.0)),
                    screw_length_mm=float(item.get("screw_length_mm", 0.0)),
                    cube_side_mm=float(item.get("cube_side_mm", 0.0)),
                    grasp_feature=str(item.get("grasp_feature", "center")),
                )
            )
        ordered_targets = sorted(
            targets,
            key=lambda target: (
                class_order[target.sort_class],
                np.hypot(*target.object_xy_mm),
            ),
        )
        self._validate_separation(ordered_targets)
        return ordered_targets
