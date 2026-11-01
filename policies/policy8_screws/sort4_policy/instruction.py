"""Parse Policy 8's dedicated two-bin, variable-count screw task."""

from __future__ import annotations

import re
from dataclasses import dataclass


CLASS_TO_BIN = {
    "m4_screw": "red",
    "m8_screw": "green",
}
SUPPORTED_BIN_COLORS = ("red", "green", "yellow")


@dataclass(frozen=True)
class Sort4Task:
    """Legacy class name retained to keep the proven controller API stable."""

    instruction: str
    class_to_bin: dict[str, str]


def default_task(instruction: str = "") -> Sort4Task:
    if str(instruction).strip():
        return parse_instruction(instruction)
    return Sort4Task(
        instruction="",
        class_to_bin=dict(CLASS_TO_BIN),
    )


def _destination(normalized: str, object_pattern: str, label: str) -> str:
    colors = "|".join(SUPPORTED_BIN_COLORS)
    matches = set(
        re.findall(
            rf"{object_pattern}.{{0,80}}?\b(?:into|to|in)\s+(?:the\s+)?"
            rf"({colors})\s+(?:bin|box)\b",
            normalized,
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Policy 8 prompt must specify exactly one destination for {label}")
    return matches.pop()


def parse_instruction(text: str) -> Sort4Task:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        raise ValueError("The rollout instruction is empty")
    if re.search(r"\b(cubes?|pills?|tablets?|batter(?:y|ies))\b", normalized):
        raise ValueError("Policy 8 only supports M4 and M8 screws")
    mapping = {
        "m4_screw": _destination(normalized, r"\bm4\s+screws?\b", "M4 screws"),
        "m8_screw": _destination(normalized, r"\bm8\s+screws?\b", "M8 screws"),
    }
    if len(set(mapping.values())) != 2:
        raise ValueError("Policy 8 prompt must assign M4 and M8 to different supported bins")
    return Sort4Task(normalized, mapping)
