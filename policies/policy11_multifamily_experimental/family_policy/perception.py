"""Composite, metric-authoritative perception for Policy 11."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from battery_policy.instruction import SortTask as BatteryTask
from battery_policy.perception import (
    CameraBlockedError as BatteryCameraBlocked,
    PerceptionError as BatteryPerceptionError,
    SortTarget as BatteryTarget,
    TrainedDetectorClient as BatteryClient,
)
from family_policy.bin_profiles import BIN_PROFILES
from family_policy.config import (
    CLASS_FAMILY,
    FAMILY_CLASSES,
    MAX_TASK_OBJECTS,
    MIN_CROSS_FAMILY_SEPARATION_MM,
)
from family_policy.instruction import SortTask
from pill_policy.instruction import SortTask as PillTask
from pill_policy.perception import (
    CameraBlockedError as PillCameraBlocked,
    PerceptionError as PillPerceptionError,
    SortTarget as PillTarget,
    TrainedDetectorClient as PillClient,
)
from sort4_policy.instruction import Sort4Task as ScrewTask
from sort4_policy.perception import (
    PerceptionError as ScrewPerceptionError,
    Sort4CosmosClient as ScrewClient,
    SortTarget as ScrewTarget,
)


class PerceptionError(RuntimeError):
    pass


class CameraBlockedError(PerceptionError):
    pass


@dataclass(frozen=True)
class SortTarget:
    sort_class: str
    label: str
    object_xy_mm: tuple[float, float]
    object_yaw_deg: float
    bin_color: str
    bin_xy_mm: tuple[float, float]
    bin_profile: str
    bin_footprint_mm: float
    bin_body_height_mm: float
    bin_release_z_mm: float
    source_family: str
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
    object_length_mm: float = 0.0
    object_width_mm: float = 0.0
    model_confidence: float = 0.0
    grasp_feature: str = "center"


class MultiFamilyDetectorClient:
    """Route each requested family through its proven specialized verifier."""

    _POSITIONS = {
        "m4_screw": [0.34, 0.58],
        "m8_screw": [0.45, 0.58],
        "aaa_battery": [0.56, 0.58],
        "aa_battery": [0.66, 0.58],
        "round_pill": [0.39, 0.70],
        "long_pill": [0.57, 0.70],
    }

    def __init__(self):
        self.clients = {
            "screw": ScrewClient(),
            "battery": BatteryClient(),
            "pill": PillClient(),
        }
        self.active_task: SortTask | None = None
        self.family_tasks: dict[str, object] = {}

    @staticmethod
    def _task_for_family(family: str, task: SortTask):
        mapping = {
            name: task.class_to_bin[name]
            for name in FAMILY_CLASSES[family]
            if name in task.class_to_bin
        }
        if not mapping:
            return None
        task_types = {
            "screw": ScrewTask,
            "battery": BatteryTask,
            "pill": PillTask,
        }
        return task_types[family](task.instruction, mapping)

    def _activate(self, task: SortTask) -> None:
        self.active_task = task
        self.family_tasks = {
            family: family_task
            for family in FAMILY_CLASSES
            if (family_task := self._task_for_family(family, task)) is not None
        }

    @staticmethod
    def _unified(target) -> SortTarget:
        sort_class = str(target.sort_class)
        values = {
            field.name: getattr(target, field.name, field.default)
            for field in fields(SortTarget)
            if field.name != "source_family"
        }
        values["source_family"] = CLASS_FAMILY[sort_class]
        return SortTarget(**values)

    @staticmethod
    def _vendor_target(target: SortTarget, family: str):
        target_types = {
            "screw": ScrewTarget,
            "battery": BatteryTarget,
            "pill": PillTarget,
        }
        target_type = target_types[family]
        return target_type(
            **{
                field.name: getattr(target, field.name)
                for field in fields(target_type)
            }
        )

    def _sort_and_validate(
        self, targets: list[SortTarget], task: SortTask, *, allow_empty: bool
    ) -> list[SortTarget]:
        if not targets and not allow_empty:
            raise PerceptionError("No prompt-selected objects were detected")
        if len(targets) > MAX_TASK_OBJECTS:
            raise PerceptionError(
                f"Dynamic inventory exceeds the {MAX_TASK_OBJECTS}-object safety ceiling"
            )
        for index, left in enumerate(targets):
            for right in targets[index + 1 :]:
                separation = float(
                    np.linalg.norm(
                        np.asarray(left.object_xy_mm) - np.asarray(right.object_xy_mm)
                    )
                )
                if separation < MIN_CROSS_FAMILY_SEPARATION_MM:
                    raise PerceptionError(
                        f"Objects overlap or are cross-family ambiguous: {separation:.1f} mm"
                    )
        order = {name: index for index, name in enumerate(task.class_to_bin)}
        return sorted(
            targets,
            key=lambda item: (order[item.sort_class], np.hypot(*item.object_xy_mm)),
        )

    @staticmethod
    def _raise_family_error(family: str, exc: Exception):
        if isinstance(exc, (BatteryCameraBlocked, PillCameraBlocked)):
            raise CameraBlockedError(str(exc)) from exc
        raise PerceptionError(f"{family} perception failed: {exc}") from exc

    def mock_scene(self, task: SortTask) -> dict:
        self._activate(task)
        profile_name = "old" if "yellow" in task.class_to_bin.values() else "new"
        profile = BIN_PROFILES[profile_name]
        targets: list[SortTarget] = []
        for family, family_task in self.family_tasks.items():
            client = self.clients[family]
            scene = client.mock_scene(family_task)
            selected = []
            seen = set()
            for raw in scene.get("objects", []):
                sort_class = raw.get("sort_class")
                if sort_class not in family_task.class_to_bin or sort_class in seen:
                    continue
                seen.add(sort_class)
                item = dict(raw)
                center = self._POSITIONS[sort_class]
                item["center"] = center
                item["grasp_center"] = center
                if item.get("axis_endpoints") is not None:
                    item["axis_endpoints"] = [
                        [center[0] - 0.015, center[1]],
                        [center[0] + 0.015, center[1]],
                    ]
                selected.append(item)
            scene["objects"] = selected
            for bin_item in scene.get("bins", []):
                bin_item.update(
                    {
                        "profile": profile_name,
                        "footprint_mm": profile["footprint_mm"],
                        "body_height_m": profile["body_height_m"],
                        "visual_midpoint_m": profile["visual_midpoint_m"],
                        "release_z_mm": profile["release_z_mm"],
                    }
                )
            targets.extend(
                self._unified(item)
                for item in client.to_targets(scene, family_task)
            )
        return {
            "status": "PASS",
            "targets": self._sort_and_validate(targets, task, allow_empty=False),
            "verification": "mock composite inventory",
        }

    def infer(self, image_rgb: np.ndarray, task: SortTask) -> dict:
        self._activate(task)
        targets: list[SortTarget] = []
        for family, family_task in self.family_tasks.items():
            client = self.clients[family]
            try:
                scene = client.infer(image_rgb, family_task)
                family_targets = client.to_targets(scene, family_task)
            except (
                ScrewPerceptionError,
                BatteryPerceptionError,
                PillPerceptionError,
            ) as exc:
                self._raise_family_error(family, exc)
            targets.extend(self._unified(item) for item in family_targets)
        return {
            "status": "PASS",
            "targets": self._sort_and_validate(targets, task, allow_empty=False),
            "verification": "specialized screw + battery + pill metric verification",
        }

    def to_targets(self, scene: dict, task: SortTask) -> list[SortTarget]:
        targets = list(scene.get("targets", []))
        if not all(isinstance(item, SortTarget) for item in targets):
            raise PerceptionError("Composite scene contains invalid targets")
        return self._sort_and_validate(targets, task, allow_empty=False)

    def relocalize_targets(
        self, image_rgb: np.ndarray, old_targets: list[SortTarget]
    ) -> list[SortTarget]:
        if not old_targets:
            return []
        if self.active_task is None:
            raise PerceptionError("No active Policy 11 task")
        refreshed_by_index: dict[int, SortTarget] = {}
        for family in FAMILY_CLASSES:
            indexed = [
                (index, target)
                for index, target in enumerate(old_targets)
                if target.source_family == family
            ]
            if not indexed:
                continue
            client = self.clients[family]
            try:
                refreshed = client.relocalize_targets(
                    image_rgb,
                    [self._vendor_target(target, family) for _, target in indexed],
                )
            except (
                ScrewPerceptionError,
                BatteryPerceptionError,
                PillPerceptionError,
            ) as exc:
                self._raise_family_error(family, exc)
            if len(refreshed) != len(indexed):
                raise PerceptionError(
                    f"{family} relocalization changed target count "
                    f"{len(indexed)} -> {len(refreshed)}"
                )
            for (index, _), target in zip(indexed, refreshed):
                refreshed_by_index[index] = self._unified(target)
        result = [refreshed_by_index[index] for index in range(len(old_targets))]
        return self._sort_and_validate(result, self.active_task, allow_empty=False)

    def target_is_present(self, image_rgb: np.ndarray, target: SortTarget) -> bool:
        family = target.source_family
        try:
            return bool(
                self.clients[family].target_is_present(
                    image_rgb, self._vendor_target(target, family)
                )
            )
        except (
            ScrewPerceptionError,
            BatteryPerceptionError,
            PillPerceptionError,
        ) as exc:
            self._raise_family_error(family, exc)

    def residual_targets(
        self, image_rgb: np.ndarray, task: SortTask
    ) -> list[SortTarget]:
        if self.active_task is None:
            self._activate(task)
        targets: list[SortTarget] = []
        for family, family_task in self.family_tasks.items():
            try:
                family_targets = self.clients[family].residual_targets(
                    image_rgb, family_task
                )
            except (
                ScrewPerceptionError,
                BatteryPerceptionError,
                PillPerceptionError,
            ) as exc:
                self._raise_family_error(family, exc)
            targets.extend(self._unified(item) for item in family_targets)
        return self._sort_and_validate(targets, task, allow_empty=True)


# Keep the proven generic controller import stable.
TrainedDetectorClient = MultiFamilyDetectorClient
