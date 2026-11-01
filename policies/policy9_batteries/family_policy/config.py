"""Immutable task contract for Astra Policy 9."""

from __future__ import annotations

import os

from family_policy.bin_profiles import BIN_PROFILES, DEFAULT_BIN_PROFILE


POLICY_NUMBER = 9
POLICY_NAME = "variable-count AA/AAA battery sorting"
FAMILY = "battery"
CLASS_TO_BIN = {
    "aaa_battery": "red",    # smaller
    "aa_battery": "green",  # larger
}
CLASS_LABELS = {
    "aaa_battery": "AAA battery",
    "aa_battery": "AA battery",
}
OBJECT_HEIGHT_M = {
    "aaa_battery": 0.0051,
    "aa_battery": 0.0070,
}
# Ranges include perspective/segmentation tolerance around the measured CAD
# (AAA 44.0 x 10.2 mm, AA 49.0 x 14.0 mm) and the first real CSI frame.
# Length is the cleanest separator after the real black/gold cells showed that
# synthetic colour is not reliable.  The small gap fails closed at the class
# boundary instead of silently swapping AA and AAA.
GEOMETRY = {
    "aaa_battery": {"length": (35.0, 48.9), "width": (7.0, 17.0), "aspect": (2.3, 7.5)},
    "aa_battery": {"length": (49.0, 65.0), "width": (11.0, 24.5), "aspect": (2.1, 6.5)},
}
VERIFY_CLASS_BY_GEOMETRY = True
# Use one observation plane while deciding the class so an incorrect model
# label cannot change the pixel-to-metric scale used to override that label.
METRIC_CLASSIFICATION_HEIGHT_M = 0.006
MIN_SEPARATION_MM = 20.0
MAX_TASK_OBJECTS = int(os.environ.get("DEMO_MAX_BATTERIES", "24"))
BIN_BODY_HEIGHT_M = BIN_PROFILES[DEFAULT_BIN_PROFILE]["body_height_m"]
BIN_VISUAL_MIDPOINT_M = BIN_BODY_HEIGHT_M / 2.0
MODEL_CLASSES = tuple(CLASS_TO_BIN)
DEFAULT_INSTRUCTION = (
    "Sort all AAA batteries into the red bin and all AA batteries into the green bin."
)
GRASP_Z_MM = {name: float(os.environ.get("DEMO_BATTERY_GRASP_Z_MM", "-334")) for name in CLASS_TO_BIN}
CHECKPOINT = os.environ.get(
    "ASTRA_DETECTOR_CHECKPOINT",
    "/workspace/real_object_classifier_20261031/training/run/weights/best.pt",
)
DETECTOR_PORT = int(os.environ.get("ASTRA_DETECTOR_PORT", "18766"))
COSMOS_URL = os.environ.get("COSMOS_DEMO_URL", "http://127.0.0.1:18765")
