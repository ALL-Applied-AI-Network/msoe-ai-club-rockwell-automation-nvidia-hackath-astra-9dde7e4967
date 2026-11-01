"""Task and safety contract for Astra Policy 11."""

from __future__ import annotations

import os


POLICY_NUMBER = 11
POLICY_NAME = "dynamic multi-family object sorting"

FAMILY_CLASSES = {
    "screw": ("m4_screw", "m8_screw"),
    "battery": ("aaa_battery", "aa_battery"),
    "pill": ("round_pill", "long_pill"),
}
CLASS_FAMILY = {
    sort_class: family
    for family, sort_classes in FAMILY_CLASSES.items()
    for sort_class in sort_classes
}
CLASS_LABELS = {
    "m4_screw": "M4 screw",
    "m8_screw": "M8 screw",
    "aaa_battery": "AAA battery",
    "aa_battery": "AA battery",
    "round_pill": "round pill",
    "long_pill": "long pill",
}

# Used only by the default submission/mock. Runtime routing comes from the
# instruction delivered in obs["trial"]["instruction"].
CLASS_TO_BIN = {
    "m4_screw": "red",
    "m8_screw": "red",
    "aaa_battery": "green",
    "aa_battery": "green",
    "round_pill": "yellow",
    "long_pill": "yellow",
}
DEFAULT_INSTRUCTION = (
    "Sort all screws into the red bin, all batteries into the green bin, "
    "and all pills into the yellow bin."
)

SUPPORTED_BIN_COLORS = ("red", "green", "yellow", "blue")
MODEL_CLASSES = tuple(CLASS_LABELS)
MAX_TASK_OBJECTS = int(os.environ.get("DEMO_MAX_POLICY11_OBJECTS", "48"))
MIN_CROSS_FAMILY_SEPARATION_MM = float(
    os.environ.get("DEMO_POLICY11_MIN_SEPARATION_MM", "18")
)
GRASP_Z_MM = {
    name: float(os.environ.get("DEMO_POLICY11_GRASP_Z_MM", "-334"))
    for name in CLASS_LABELS
}

CHECKPOINT = os.environ.get(
    "ASTRA_DETECTOR_CHECKPOINT",
    "/workspace/real_object_classifier_20261031/training/run/weights/best.pt",
)
DETECTOR_PORT = int(os.environ.get("ASTRA_DETECTOR_PORT", "18766"))
COSMOS_URL = os.environ.get("COSMOS_DEMO_URL", "http://127.0.0.1:18765")
