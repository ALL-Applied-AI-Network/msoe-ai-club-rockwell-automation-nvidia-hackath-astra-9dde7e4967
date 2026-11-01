"""Guarded controller for Astra's variable-count two-class sorting policy."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from family_policy.instruction import parse_instruction
from family_policy.perception import (
    CameraBlockedError,
    PerceptionError,
    TrainedDetectorClient,
)
from family_policy.trajectory import dwell, interpolate, safe_pose
from family_policy.config import GRASP_Z_MM, POLICY_NUMBER


# CSI's recorded episode-start TCP pose, rounded to stable command values.
SAFE_HOME = np.array([300.0, 0.0, -90.0, -180.0, 0.0, 0.0, 0.0])
APPROACH_Z = float(os.environ.get("DEMO_APPROACH_Z_MM", "-245"))
LIFT_Z = float(os.environ.get("DEMO_LIFT_Z_MM", "-225"))
BIN_RELEASE_Z = float(os.environ.get("DEMO_BIN_RELEASE_Z_MM", "-265"))
GRASP_Z = GRASP_Z_MM
STABLE_XY_MM = 2.0
STABLE_YAW_DEG = 4.0
STABLE_BIN_MM = 2.0


class VariableSortController:
    def __init__(self):
        self.vision = TrainedDetectorClient()
        self.log_root = Path(
            os.environ.get(
                "DEMO_LOG_DIR",
                str(Path(__file__).resolve().parents[1] / "runtime_logs"),
            )
        )
        session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.session_root = self.log_root / f"policy{POLICY_NUMBER}_{session}_{os.getpid()}"
        self.reset()

    def reset(self) -> None:
        self.phase = "scan"
        self.task = None
        self.target = None
        self.pending_targets = []
        self.completed = Counter()
        self.relocalization_failures = 0
        self.initial_targets = []
        self.initial_inventory = Counter()
        self.last_placed_target = None
        self.pick_retry_count = 0
        self.clean_source_frames = 0
        self.residual_previous = []
        self.last_command = SAFE_HOME.copy()
        self.log_path = None
        self.frame_dir = None
        self.frame_index = 0

    @staticmethod
    def _is_mock(obs) -> bool:
        return str(((obs.get("trial") or {}).get("trial_id")) or "").startswith(
            "mock-"
        )

    def _ensure_log(self, obs) -> None:
        if self.log_path is not None:
            return
        trial = str(((obs.get("trial") or {}).get("trial_id")) or f"local-{int(time.time())}")
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.session_root / f"{trial}.jsonl"
        self.frame_dir = self.session_root / f"{trial}_frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    def _capture(self, obs) -> None:
        image = (obs.get("images") or {}).get("table")
        if image is None:
            return
        path = self.frame_dir / f"frame_{self.frame_index:05d}.png"
        bgr = cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Could not write runtime frame {path}")
        self.frame_index += 1

    def _log(self, event: str, **fields) -> None:
        row = {"time": time.time(), "event": event, "phase": self.phase, **fields}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=float) + "\n")

    def _real_gate(self, obs) -> None:
        if self._is_mock(obs):
            return
        required = (
            "DEMO_ALLOW_REAL_ROBOT",
            "DEMO_UTOOL1_CONFIRMED",
            "DEMO_TOUCH_OFF_CONFIRMED",
        )
        missing = [key for key in required if os.environ.get(key) != "YES"]
        if missing:
            raise RuntimeError(
                "Real-robot motion is locked; missing confirmations: "
                + ", ".join(missing)
            )

    def _current(self, obs) -> np.ndarray:
        tcp = obs.get("tcp_pose")
        if tcp is None:
            result = self.last_command.copy()
        else:
            value = np.asarray(tcp, dtype=float)
            result = np.concatenate((value[:6], [float(obs["joints"][-1] >= 0.5)]))
        return safe_pose(result)

    def _move(self, obs, destination, next_phase):
        destination = safe_pose(destination)
        rows = interpolate(self._current(obs), destination)
        self.last_command = destination
        self._log(
            "motion",
            destination=destination.tolist(),
            rows=len(rows),
            next_phase=next_phase,
        )
        self.phase = next_phase
        return rows

    def _hold(self, obs, seconds=1.0):
        current = self._current(obs)
        self.last_command = current
        return dwell(current, seconds)

    @staticmethod
    def _validate_target(target) -> None:
        x, y = target.object_xy_mm
        bx, by = target.bin_xy_mm
        yaw = target.object_yaw_deg
        grasp_z = GRASP_Z[target.sort_class]
        bin_release_z = float(target.bin_release_z_mm)
        for pose in (
            [x, y, APPROACH_Z, -180, 0, yaw, 0],
            [x, y, grasp_z, -180, 0, yaw, 0],
            [x, y, LIFT_Z, -180, 0, yaw, 1],
            [bx, by, LIFT_Z, -180, 0, 0, 1],
            [bx, by, bin_release_z, -180, 0, 0, 1],
        ):
            safe_pose(pose)

    @staticmethod
    def _validate_stability(targets, require_bin_stability=True) -> None:
        for target in targets:
            if target.relocalization_mm > STABLE_XY_MM:
                raise PerceptionError(
                    f"{target.label} centering changed "
                    f"{target.relocalization_mm:.2f} mm between frames"
                )
            if target.relocalization_yaw_deg > STABLE_YAW_DEG:
                raise PerceptionError(
                    f"{target.label} yaw changed "
                    f"{target.relocalization_yaw_deg:.2f} degrees between frames"
                )
            if require_bin_stability and target.bin_relocalization_mm > STABLE_BIN_MM:
                raise PerceptionError(
                    f"{target.bin_color} bin changed "
                    f"{target.bin_relocalization_mm:.2f} mm between frames"
                )

    @staticmethod
    def _same_targets(previous, current, tolerance_mm=2.0) -> bool:
        if len(previous) != len(current) or not current:
            return False
        return all(
            old.sort_class == new.sort_class
            and float(
                np.linalg.norm(
                    np.asarray(old.object_xy_mm)
                    - np.asarray(new.object_xy_mm)
                )
            )
            <= tolerance_mm
            for old, new in zip(previous, current)
        )

    def act(self, obs: dict):
        self._ensure_log(obs)
        self._capture(obs)
        self._real_gate(obs)
        instruction = str((obs.get("trial") or {}).get("instruction") or "")
        normalized_instruction = " ".join(instruction.lower().split())
        if self.task is None:
            # The cell-delivered instruction is the only runtime mapping
            # authority. Mock and real trials deliberately use the same parser.
            self.task = parse_instruction(instruction)
            self._log("instruction", instruction=instruction, mapping=self.task.class_to_bin)
        elif normalized_instruction != getattr(self.task, "instruction", normalized_instruction):
            self.phase = "done"
            self._log(
                "instruction_changed",
                planned_instruction=getattr(self.task, "instruction", ""),
                received_instruction=instruction,
            )
            return self._hold(obs)

        if self.phase == "scan":
            try:
                scene = (
                    self.vision.mock_scene(self.task)
                    if self._is_mock(obs)
                    else self.vision.infer(obs["images"]["table"], self.task)
                )
                targets = self.vision.to_targets(scene, self.task)
                for target in targets:
                    self._validate_target(target)
            except CameraBlockedError as exc:
                self.relocalization_failures += 1
                self._log(
                    "camera_blocked_rescan",
                    error=str(exc),
                    attempt=self.relocalization_failures,
                    destination=SAFE_HOME.tolist(),
                )
                if self.relocalization_failures >= 3:
                    self.phase = "done"
                    self._log("abort", reason="Initial camera-clear scan failed three times")
                    return self._hold(obs)
                return self._move(obs, SAFE_HOME, "scan")
            except PerceptionError as exc:
                self.phase = "done"
                self._log("perception_failure", error=str(exc))
                return self._hold(obs)
            self._log(
                "batch_plan",
                inventory=dict(Counter(t.sort_class for t in targets)),
                ignored_objects=scene.get("ignored_objects", []),
                targets=[
                    {
                        "class": t.sort_class,
                        "label": t.label,
                        "object_xy_mm": t.object_xy_mm,
                        "yaw_deg": t.object_yaw_deg,
                        "bin": t.bin_color,
                        "bin_xy_mm": t.bin_xy_mm,
                        "bin_profile": t.bin_profile,
                        "bin_footprint_mm": t.bin_footprint_mm,
                        "bin_body_height_mm": t.bin_body_height_mm,
                        "bin_release_z_mm": t.bin_release_z_mm,
                        "grasp_feature": t.grasp_feature,
                        "object_length_mm": t.object_length_mm,
                        "object_width_mm": t.object_width_mm,
                        "model_confidence": t.model_confidence,
                    }
                    for t in targets
                ],
            )
            self.initial_inventory = Counter(t.sort_class for t in targets)
            if not self._is_mock(obs):
                self.target = None
                self.pending_targets = list(targets)
                self.phase = "confirm_plan"
                return self._hold(obs, 1.0)
            self.target, self.pending_targets = targets[0], targets[1:]
            x, y = self.target.object_xy_mm
            return self._move(
                obs,
                [x, y, APPROACH_Z, -180, 0, self.target.object_yaw_deg, 0],
                "descend_pick",
            )

        if self.phase == "confirm_plan":
            try:
                confirmed = self.vision.relocalize_targets(
                    obs["images"]["table"], self.pending_targets
                )
                self._validate_stability(confirmed)
                for target in confirmed:
                    self._validate_target(target)
            except (PerceptionError, ValueError) as exc:
                self.relocalization_failures += 1
                self._log(
                    "initial_confirmation_failure",
                    error=str(exc),
                    attempt=self.relocalization_failures,
                )
                if self.relocalization_failures >= 3:
                    self.phase = "done"
                    self._log(
                        "abort",
                        reason=f"Policy {POLICY_NUMBER} initial two-frame confirmation failed",
                    )
                return self._hold(obs, 1.0)
            self.relocalization_failures = 0
            self.initial_targets = list(confirmed)
            confirmed_inventory = Counter(item.sort_class for item in confirmed)
            if confirmed_inventory != self.initial_inventory:
                self.phase = "done"
                self._log(
                    "abort",
                    reason="Initial dynamic inventory changed during confirmation",
                    planned=dict(self.initial_inventory),
                    confirmed=dict(confirmed_inventory),
                )
                return self._hold(obs, 1.0)
            self.target, self.pending_targets = confirmed[0], confirmed[1:]
            self._log(
                "initial_plan_confirmed",
                targets=len(confirmed),
                max_shift_mm=max(item.relocalization_mm for item in confirmed),
                max_yaw_shift_deg=max(
                    item.relocalization_yaw_deg for item in confirmed
                ),
                grasp_features=[item.grasp_feature for item in confirmed],
            )
            x, y = self.target.object_xy_mm
            return self._move(
                obs,
                [x, y, APPROACH_Z, -180, 0, self.target.object_yaw_deg, 0],
                "descend_pick",
            )

        if self.phase == "relocalize":
            try:
                refreshed = self.vision.relocalize_targets(
                    obs["images"]["table"], self.pending_targets
                )
                if not refreshed:
                    raise PerceptionError(f"No remaining Policy {POLICY_NUMBER} target to relocalize")
                for target in refreshed:
                    self._validate_target(target)
            except PerceptionError as exc:
                self.relocalization_failures += 1
                self._log(
                    "relocalization_failure",
                    error=str(exc),
                    attempt=self.relocalization_failures,
                )
                if self.relocalization_failures >= 3:
                    self.phase = "done"
                    self._log(
                        "abort",
                        reason=f"Policy {POLICY_NUMBER} remaining-target relocalization failed closed",
                        completed=dict(self.completed),
                    )
                return self._hold(obs)
            self.relocalization_failures = 0
            self.target, self.pending_targets = refreshed[0], refreshed[1:]
            self._log(
                "relocalization",
                remaining=1 + len(self.pending_targets),
                targets=[
                    {
                        "class": item.sort_class,
                        "label": item.label,
                        "object_xy_mm": item.object_xy_mm,
                        "yaw_deg": item.object_yaw_deg,
                        "shift_mm": item.relocalization_mm,
                        "yaw_shift_deg": item.relocalization_yaw_deg,
                        "bin_shift_mm": item.bin_relocalization_mm,
                    }
                    for item in refreshed
                ],
            )
            self.phase = "confirm_target"
            return self._hold(obs, 1.0)

        if self.phase == "confirm_target":
            try:
                confirmed = self.vision.relocalize_targets(
                    obs["images"]["table"], [self.target]
                )
                if len(confirmed) != 1:
                    raise PerceptionError("Target confirmation did not return one object")
                self._validate_stability(confirmed, require_bin_stability=False)
                self._validate_target(confirmed[0])
            except (PerceptionError, ValueError) as exc:
                self.relocalization_failures += 1
                self._log(
                    "target_confirmation_failure",
                    error=str(exc),
                    attempt=self.relocalization_failures,
                )
                if self.relocalization_failures >= 3:
                    self.phase = "done"
                    self._log(
                        "abort",
                        reason=f"Policy {POLICY_NUMBER} precise target centering failed",
                        completed=dict(self.completed),
                    )
                return self._hold(obs, 1.0)
            self.relocalization_failures = 0
            self.target = confirmed[0]
            self._log(
                "target_center_confirmed",
                sort_class=self.target.sort_class,
                object_xy_mm=self.target.object_xy_mm,
                shift_mm=self.target.relocalization_mm,
                yaw_shift_deg=self.target.relocalization_yaw_deg,
                grasp_feature=self.target.grasp_feature,
            )
            x, y = self.target.object_xy_mm
            return self._move(
                obs,
                [x, y, APPROACH_Z, -180, 0, self.target.object_yaw_deg, 0],
                "descend_pick",
            )

        if self.phase == "verify_pick":
            try:
                if self.last_placed_target is None:
                    raise PerceptionError("No released target to verify")
                still_present = self.vision.target_is_present(
                    obs["images"]["table"], self.last_placed_target
                )
                refreshed = []
                if still_present:
                    refreshed = self.vision.relocalize_targets(
                        obs["images"]["table"], [self.last_placed_target]
                    )
                    if len(refreshed) != 1:
                        raise PerceptionError(
                            "Pickup retry did not return exactly one target"
                        )
                    self._validate_target(refreshed[0])
            except (PerceptionError, ValueError) as exc:
                self.relocalization_failures += 1
                self._log(
                    "pick_verification_failure",
                    error=str(exc),
                    attempt=self.relocalization_failures,
                )
                if self.relocalization_failures >= 3:
                    self.phase = "done"
                    self._log(
                        "abort",
                        reason=f"Policy {POLICY_NUMBER} pickup verification failed closed",
                        completed=dict(self.completed),
                    )
                return self._hold(obs, 1.0)
            self.relocalization_failures = 0
            if still_present:
                if self.pick_retry_count >= 1:
                    self.phase = "done"
                    self._log(
                        "abort",
                        reason="Object remained at source after one pickup retry",
                        sort_class=self.last_placed_target.sort_class,
                        object_xy_mm=self.last_placed_target.object_xy_mm,
                        completed=dict(self.completed),
                    )
                    return self._hold(obs, 1.0)
                self.pick_retry_count += 1
                self.target = refreshed[0]
                self._log(
                    "pickup_retry",
                    sort_class=self.target.sort_class,
                    object_xy_mm=self.target.object_xy_mm,
                    retry=self.pick_retry_count,
                )
                self._log(
                    "camera_clear_relocalization_pose",
                    destination=SAFE_HOME.tolist(),
                    next_phase="confirm_target",
                )
                return self._move(obs, SAFE_HOME, "confirm_target")

            removed = self.last_placed_target
            self.completed[removed.sort_class] += 1
            self.last_placed_target = None
            self.pick_retry_count = 0
            self._log(
                "pickup_verified",
                sort_class=removed.sort_class,
                completed=dict(self.completed),
            )
            if self.pending_targets:
                self.target = None
                next_phase = "relocalize"
            else:
                self.target = None
                self.clean_source_frames = 0
                self.residual_previous = []
                next_phase = "final_audit"
            self._log(
                "camera_clear_relocalization_pose",
                destination=SAFE_HOME.tolist(),
                next_phase=next_phase,
            )
            return self._move(obs, SAFE_HOME, next_phase)

        if self.phase == "final_audit":
            try:
                residual = self.vision.residual_targets(
                    obs["images"]["table"], self.task
                )
                for target in residual:
                    self._validate_target(target)
            except (PerceptionError, ValueError) as exc:
                self.relocalization_failures += 1
                self.residual_previous = []
                self.clean_source_frames = 0
                self._log(
                    "final_audit_failure",
                    error=str(exc),
                    attempt=self.relocalization_failures,
                )
                if isinstance(exc, CameraBlockedError) and self.relocalization_failures < 3:
                    self._log(
                        "camera_clear_final_audit_pose",
                        destination=SAFE_HOME.tolist(),
                    )
                    return self._move(obs, SAFE_HOME, "final_audit")
                if self.relocalization_failures >= 3:
                    self.phase = "done"
                    self._log(
                        "abort",
                        reason=f"Policy {POLICY_NUMBER} final source audit failed closed",
                        completed=dict(self.completed),
                    )
                return self._hold(obs, 1.0)
            self.relocalization_failures = 0
            if not residual:
                self.residual_previous = []
                self.clean_source_frames += 1
                self._log(
                    "final_source_audit",
                    residual=0,
                    clean_frames=self.clean_source_frames,
                )
                if self.clean_source_frames < 2:
                    return self._hold(obs, 1.0)
                self._log(
                    "complete",
                    completed=dict(self.completed),
                    initial_inventory=dict(self.initial_inventory),
                    clean_source_frames=self.clean_source_frames,
                )
                return self._move(obs, SAFE_HOME, "done")

            self.clean_source_frames = 0
            stable = self._same_targets(self.residual_previous, residual)
            self.residual_previous = residual
            self._log(
                "final_source_audit",
                residual=len(residual),
                stable_frames=2 if stable else 1,
                targets=[
                    {
                        "class": item.sort_class,
                        "object_xy_mm": item.object_xy_mm,
                    }
                    for item in residual
                ],
            )
            if not stable:
                return self._hold(obs, 1.0)
            self.target, self.pending_targets = residual[0], residual[1:]
            self.residual_previous = []
            self._log(
                "residual_recovery_plan",
                targets=len(residual),
                completed_before_recovery=dict(self.completed),
            )
            x, y = self.target.object_xy_mm
            return self._move(
                obs,
                [x, y, APPROACH_Z, -180, 0, self.target.object_yaw_deg, 0],
                "descend_pick",
            )

        if self.phase == "done" or self.target is None:
            self.phase = "done"
            return self._hold(obs)

        x, y = self.target.object_xy_mm
        bx, by = self.target.bin_xy_mm
        yaw = self.target.object_yaw_deg
        grasp_z = GRASP_Z[self.target.sort_class]
        bin_release_z = float(self.target.bin_release_z_mm)
        if self.phase == "approach_next":
            return self._move(obs, [x, y, APPROACH_Z, -180, 0, yaw, 0], "descend_pick")
        if self.phase == "descend_pick":
            return self._move(obs, [x, y, grasp_z, -180, 0, yaw, 0], "close")
        if self.phase == "close":
            closed = safe_pose([x, y, grasp_z, -180, 0, yaw, 1])
            self.last_command = closed
            self.phase = "lift"
            self._log("gripper_close", sort_class=self.target.sort_class)
            return dwell(closed, 0.8)
        if self.phase == "lift":
            return self._move(obs, [x, y, LIFT_Z, -180, 0, yaw, 1], "above_bin")
        if self.phase == "above_bin":
            return self._move(obs, [bx, by, LIFT_Z, -180, 0, 0, 1], "lower_bin")
        if self.phase == "lower_bin":
            return self._move(obs, [bx, by, bin_release_z, -180, 0, 0, 1], "open")
        if self.phase == "open":
            opened = safe_pose([bx, by, bin_release_z, -180, 0, 0, 0])
            self.last_command = opened
            self.phase = "retreat"
            if self._is_mock(obs):
                self.completed[self.target.sort_class] += 1
            self._log(
                "gripper_open",
                pending_pick_verification=not self._is_mock(obs),
                completed=dict(self.completed),
            )
            return dwell(opened, 0.8)
        if self.phase == "retreat":
            # Refresh from the above-bin retreat pose instead of SAFE_HOME.
            # Policy 1 physical trials t04-t06 proved that SAFE_HOME puts the
            # wrist over the loose-object field in the overhead camera and can
            # create a merged/oversized silhouette.  The retreat pose is
            # already guarded and leaves the source field clear. The policy's
            # object and two-bin checks remain fail-closed if a required
            # component is not visible from this pose.
            if not self._is_mock(obs):
                self.last_placed_target = self.target
                self.target = None
                next_phase = "verify_pick"
                self._log(
                    "camera_clear_pick_verification_pose",
                    destination=[bx, by, LIFT_Z, -180, 0, 0, 0],
                )
            else:
                next_phase = "return_home"
            return self._move(
                obs,
                [bx, by, LIFT_Z, -180, 0, 0, 0],
                next_phase,
            )
        if self.phase == "return_home":
            if self.pending_targets:
                if self._is_mock(obs):
                    self.target = self.pending_targets.pop(0)
                    next_phase = "approach_next"
                else:
                    self.target = None
                    next_phase = "relocalize"
            else:
                self.target = None
                next_phase = "done"
                self._log("complete", completed=dict(self.completed))
            return self._move(obs, SAFE_HOME, next_phase)
        self.phase = "done"
        return self._hold(obs)
