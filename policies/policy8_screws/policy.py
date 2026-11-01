"""Astra Policy 8: variable-count M4/M8 screw sorting into two bins."""

from __future__ import annotations

from sort4_policy.controller import Sort4Controller


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
        self.controller = Sort4Controller()

    def reset(self) -> None:
        self.controller.reset()

    def act(self, obs: dict):
        return self.controller.act(obs)
