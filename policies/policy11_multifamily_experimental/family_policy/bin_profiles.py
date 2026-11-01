"""Calibrated old/new destination-bin profiles shared by perception and motion."""

from __future__ import annotations

import os

import numpy as np

from calibration.geometry import SURFACE_Z, pixel_to_base


# The old ManipBench bin is 50 x 50 x 55 mm.  Astra's new printed bin is
# 90 x 90 x 60 mm.  Release depths preserve the previously validated
# clearance above each rim.
BIN_PROFILES = {
    "old": {
        "footprint_mm": 50.0,
        "body_height_m": 0.055,
        "visual_midpoint_m": 0.0275,
        "release_z_mm": float(os.environ.get("DEMO_OLD_BIN_RELEASE_Z_MM", "-270")),
        "initial_span_range_mm": (65.0, 104.0),
        "tracked_span_range_mm": (55.0, 112.0),
    },
    "new": {
        "footprint_mm": 90.0,
        "body_height_m": 0.060,
        "visual_midpoint_m": 0.0300,
        "release_z_mm": float(
            os.environ.get(
                "DEMO_NEW_BIN_RELEASE_Z_MM",
                os.environ.get("DEMO_BIN_RELEASE_Z_MM", "-265"),
            )
        ),
        "initial_span_range_mm": (106.0, 180.0),
        "tracked_span_range_mm": (90.0, 195.0),
    },
}
DEFAULT_BIN_PROFILE = "new"
SUPPORTED_BIN_COLORS = ("red", "green", "yellow", "blue")
_MEASUREMENT_HEIGHT_M = 0.02875


def metric_bbox_span_mm(
    bbox_pixels: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    camera: dict,
) -> tuple[float, float, float]:
    """Return calibrated horizontal, vertical, and equivalent bbox spans."""

    x0, y0, x1, y1 = (float(value) for value in bbox_pixels)
    height, width = image_shape[:2]
    native_width, native_height = camera["size"]
    scale = np.asarray([native_width / width, native_height / height], dtype=float)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    pixels = np.asarray(
        [
            [x0, center_y],
            [x1, center_y],
            [center_x, y0],
            [center_x, y1],
        ],
        dtype=float,
    ) * scale
    base = np.asarray(
        [
            pixel_to_base(
                camera,
                pixel,
                z=SURFACE_Z + _MEASUREMENT_HEIGHT_M,
            )
            for pixel in pixels
        ]
    )
    horizontal = float(np.linalg.norm(base[1, :2] - base[0, :2]) * 1000.0)
    vertical = float(np.linalg.norm(base[3, :2] - base[2, :2]) * 1000.0)
    equivalent = float(np.sqrt(horizontal * vertical))
    return horizontal, vertical, equivalent


def classify_bin_profile(metric_span_mm: float, expected: str | None = None) -> dict:
    """Classify one clear bin, or validate evidence for its tracked profile."""

    if not np.isfinite(metric_span_mm):
        raise ValueError("Bin metric span is not finite")
    if expected is not None:
        if expected not in BIN_PROFILES:
            raise ValueError(f"Unsupported tracked bin profile: {expected}")
        low, high = BIN_PROFILES[expected]["tracked_span_range_mm"]
        if not low <= metric_span_mm <= high:
            raise ValueError(
                f"Tracked {expected} bin span {metric_span_mm:.1f} mm is outside "
                f"{low:.1f}-{high:.1f} mm"
            )
        return {"name": expected, **BIN_PROFILES[expected]}

    matches = []
    for name, profile in BIN_PROFILES.items():
        low, high = profile["initial_span_range_mm"]
        if low <= metric_span_mm <= high:
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            f"Bin span {metric_span_mm:.1f} mm is not an unambiguous old/new profile"
        )
    name = matches[0]
    return {"name": name, **BIN_PROFILES[name]}

