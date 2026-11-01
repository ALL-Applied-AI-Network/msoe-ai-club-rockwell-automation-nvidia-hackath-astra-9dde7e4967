"""Fail-closed dynamic parser for Policy 11's family and subtype rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from family_policy.config import (
    CLASS_TO_BIN,
    DEFAULT_INSTRUCTION,
    FAMILY_CLASSES,
    SUPPORTED_BIN_COLORS,
)


@dataclass(frozen=True)
class SortTask:
    instruction: str
    class_to_bin: dict[str, str]


_ALIASES = {
    "srews": "screws",
    "scerws": "screws",
    "batteris": "batteries",
    "batterys": "batteries",
    "roung": "round",
    "yello": "yellow",
    "grean": "green",
}

_CLASS_PATTERNS = {
    "m4_screw": r"\bm4\s+screws?\b",
    "m8_screw": r"\bm8\s+screws?\b",
    "aaa_battery": r"\baaa\s+batter(?:y|ies)\b",
    "aa_battery": r"\baa\s+batter(?:y|ies)\b",
    "round_pill": r"\bround\s+(?:pills?|tablets?)\b",
    "long_pill": r"\blong\s+(?:pills?|tablets?)\b",
}
_FAMILY_PATTERNS = {
    "screw": r"(?<!m4\s)(?<!m8\s)\bscrews?\b",
    "battery": r"(?<!aa\s)(?<!aaa\s)\bbatter(?:y|ies)\b",
    "pill": r"(?<!round\s)(?<!long\s)\b(?:pills?|tablets?)\b",
}


def _normalise(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    for wrong, correct in _ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(wrong)}\b", correct, normalized)
    return normalized


def _destination(normalized: str, subject_pattern: str, label: str) -> str | None:
    colors = "|".join(SUPPORTED_BIN_COLORS)
    clauses = re.split(r"\s*(?:,|;|\band\b)\s*", normalized)
    matches = set()
    for clause in clauses:
        if not re.search(subject_pattern, clause):
            continue
        matches.update(
            re.findall(
                rf"{subject_pattern}.{{0,70}}?\b(?:into|to|in)\s+(?:the\s+)?"
                rf"({colors})\s+(?:bin|box)\b",
                clause,
            )
        )
    if len(matches) > 1:
        raise ValueError(f"Policy 11 prompt gives conflicting destinations for {label}")
    return next(iter(matches), None)


def default_task(instruction: str = "") -> SortTask:
    return parse_instruction(instruction or DEFAULT_INSTRUCTION)


def parse_instruction(text: str) -> SortTask:
    normalized = _normalise(text)
    if not normalized:
        raise ValueError("The rollout instruction is empty")
    if re.search(r"\b(cubes?|blocks?|other objects?)\b", normalized):
        raise ValueError("Policy 11 only supports screws, batteries, and pills")

    mapping: dict[str, str] = {}
    for family, sort_classes in FAMILY_CLASSES.items():
        family_pattern = _FAMILY_PATTERNS[family]
        family_mentioned = re.search(family_pattern, normalized) is not None
        family_destination = _destination(normalized, family_pattern, f"all {family}s")

        subtype_destinations = {
            sort_class: _destination(
                normalized,
                _CLASS_PATTERNS[sort_class],
                sort_class.replace("_", " "),
            )
            for sort_class in sort_classes
        }
        subtype_mentioned = {
            sort_class: re.search(_CLASS_PATTERNS[sort_class], normalized) is not None
            for sort_class in sort_classes
        }

        if family_destination and any(subtype_destinations.values()):
            raise ValueError(
                f"Policy 11 prompt mixes family and subtype rules for {family}s"
            )
        if family_destination:
            mapping.update({sort_class: family_destination for sort_class in sort_classes})
            continue

        for sort_class, destination in subtype_destinations.items():
            if destination:
                mapping[sort_class] = destination
            elif subtype_mentioned[sort_class]:
                raise ValueError(
                    f"Policy 11 prompt is missing a destination for "
                    f"{sort_class.replace('_', ' ')}"
                )
        if family_mentioned and not any(subtype_destinations.values()):
            raise ValueError(f"Policy 11 prompt is missing a destination for all {family}s")

    if not mapping:
        raise ValueError("Policy 11 prompt contains no supported sorting rule")
    return SortTask(normalized, mapping)
