"""Pill verifier contract vendored for Astra Policy 11."""

from __future__ import annotations

import os

from pill_policy.bin_profiles import BIN_PROFILES, DEFAULT_BIN_PROFILE


POLICY_NUMBER = 10
POLICY_NAME = "variable-count pill size sorting"
FAMILY = "pill"
CLASS_TO_BIN = {
    "round_pill": "red",   # smaller
    "long_pill": "green", # larger/longer
}
CLASS_LABELS = {
    "round_pill": "round pill",
    "long_pill": "long pill",
}
OBJECT_HEIGHT_M = {
    "round_pill": 0.00275,
    "long_pill": 0.00360,
}
# Ranges include perspective/segmentation tolerance around the measured CAD:
# round 11.5 x 11.5 mm, long 18.7 x 8.5 mm.
GEOMETRY = {
    "round_pill": {"length": (8.0, 15.0), "width": (8.0, 15.0), "aspect": (0.80, 1.50)},
    "long_pill": {"length": (15.5, 24.0), "width": (6.0, 14.0), "aspect": (1.35, 3.50)},
}
VERIFY_CLASS_BY_GEOMETRY = True
MIN_SEPARATION_MM = 14.0
MAX_TASK_OBJECTS = int(os.environ.get("DEMO_MAX_PILLS", "24"))
BIN_BODY_HEIGHT_M = BIN_PROFILES[DEFAULT_BIN_PROFILE]["body_height_m"]
BIN_VISUAL_MIDPOINT_M = BIN_BODY_HEIGHT_M / 2.0
MODEL_CLASSES = tuple(CLASS_TO_BIN)
DEFAULT_INSTRUCTION = (
    "Sort all round pills into the red bin and all long pills into the green bin."
)
GRASP_Z_MM = {name: float(os.environ.get("DEMO_PILL_GRASP_Z_MM", "-334")) for name in CLASS_TO_BIN}
CHECKPOINT = os.environ.get(
    "ASTRA_DETECTOR_CHECKPOINT",
    "/workspace/real_object_classifier_20261031/training/run/weights/best.pt",
)
DETECTOR_PORT = int(os.environ.get("ASTRA_DETECTOR_PORT", "18766"))
