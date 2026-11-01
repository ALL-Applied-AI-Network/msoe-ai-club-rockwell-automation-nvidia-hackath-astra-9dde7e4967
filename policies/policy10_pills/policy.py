"""Astra Policy 10: variable-count pill-size sorting."""

from __future__ import annotations

from family_policy.controller import VariableSortController


class Policy:
    CAMERAS = ["table"]
    IMAGE_SIZE = (1920, 1080)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = []
    DT = 0.1

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint
        self.controller = VariableSortController()

    def reset(self) -> None:
        self.controller.reset()

    def act(self, obs: dict):
        return self.controller.act(obs)
