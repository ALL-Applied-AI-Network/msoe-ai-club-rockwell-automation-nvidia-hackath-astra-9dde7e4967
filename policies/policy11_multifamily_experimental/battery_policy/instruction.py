"""Fail-closed instruction parser for the policy's fixed two-bin mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass

from battery_policy.config import CLASS_TO_BIN, DEFAULT_INSTRUCTION, FAMILY, POLICY_NUMBER


SUPPORTED_BIN_COLORS = ("red", "green", "yellow")


@dataclass(frozen=True)
class SortTask:
    instruction: str
    class_to_bin: dict[str, str]


def default_task(instruction: str = "") -> SortTask:
    if str(instruction).strip():
        return parse_instruction(instruction)
    return SortTask(
        instruction=" ".join(DEFAULT_INSTRUCTION.lower().split()),
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
        raise ValueError(
            f"Policy {POLICY_NUMBER} prompt must specify exactly one destination for {label}"
        )
    return matches.pop()


def parse_instruction(text: str) -> SortTask:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        raise ValueError("The rollout instruction is empty")
    if re.search(r"\b(screws?|pills?|tablets?|cubes?)\b", normalized):
        raise ValueError(f"Policy {POLICY_NUMBER} only supports {FAMILY}s")
    mapping = {
        "aaa_battery": _destination(
            normalized, r"\baaa\s+batter(?:y|ies)\b", "AAA batteries"
        ),
        "aa_battery": _destination(
            normalized, r"\baa\s+batter(?:y|ies)\b", "AA batteries"
        ),
    }
    if len(set(mapping.values())) != 2:
        raise ValueError(
            f"Policy {POLICY_NUMBER} prompt must assign AA and AAA to different supported bins"
        )
    return SortTask(normalized, mapping)
