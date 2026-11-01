from __future__ import annotations

import sys
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sort4_policy.controller import (  # noqa: E402
    BIN_RELEASE_Z,
    GRASP_Z,
    SAFE_HOME,
    Sort4Controller,
)
from sort4_policy.bin_profiles import classify_bin_profile  # noqa: E402
from sort4_policy.instruction import CLASS_TO_BIN, default_task, parse_instruction  # noqa: E402
from sort4_policy.perception import (  # noqa: E402
    MAX_TASK_OBJECTS,
    BIN_BODY_HEIGHT_M,
    BIN_VISUAL_MIDPOINT_M,
    PerceptionError,
    Sort4CosmosClient,
    SortTarget,
)


INSTRUCTION = (
    "Sort all M4 screws into the red bin and all M8 screws into the green bin."
)


class InstructionTests(unittest.TestCase):
    def test_policy8_mapping(self):
        task = parse_instruction(INSTRUCTION)
        self.assertEqual(task.class_to_bin, {"m4_screw": "red", "m8_screw": "green"})
        self.assertFalse(hasattr(task, "expected_counts"))

    def test_rejects_other_object_families(self):
        with self.assertRaisesRegex(ValueError, "only supports"):
            parse_instruction(INSTRUCTION + " Put pills into the blue bin.")

    def test_requires_both_destinations(self):
        with self.assertRaisesRegex(ValueError, "destination for M8 screws"):
            parse_instruction("Sort all M4 screws into the red bin.")

    def test_robot_prompt_can_swap_destinations(self):
        task = parse_instruction(
            "Sort all M4 screws into the green bin and all M8 screws into the red bin."
        )
        self.assertEqual(task.class_to_bin, {"m4_screw": "green", "m8_screw": "red"})

    def test_rejects_same_destination_for_both_classes(self):
        with self.assertRaisesRegex(ValueError, "different supported bins"):
            parse_instruction(
                "Sort all M4 screws into the red bin and all M8 screws into the red bin."
            )

    def test_old_red_yellow_prompt_is_supported(self):
        task = parse_instruction(
            "Sort all M4 screws into the red bin and all M8 screws into the yellow bin."
        )
        self.assertEqual(task.class_to_bin, {"m4_screw": "red", "m8_screw": "yellow"})


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.client = Sort4CosmosClient("http://127.0.0.1:9")
        self.task = default_task(INSTRUCTION)

    def test_mock_scene_contains_only_screws_and_two_bins(self):
        scene = self.client.mock_scene(self.task)
        self.assertEqual(
            Counter(item["sort_class"] for item in scene["objects"]),
            Counter({"m4_screw": 3, "m8_screw": 3}),
        )
        self.assertEqual({item["color"] for item in scene["bins"]}, {"red", "green"})

    def test_variable_inventory_is_not_compared_to_three_plus_three(self):
        scene = self.client.mock_scene(self.task)
        scene["objects"] = [scene["objects"][0], *scene["objects"][3:]]
        targets = self.client.to_targets(scene, self.task)
        self.assertEqual(
            Counter(target.sort_class for target in targets),
            Counter({"m4_screw": 1, "m8_screw": 3}),
        )

    def test_metric_size_classification(self):
        classify = self.client._screw_class_from_metric_shape
        self.assertEqual(classify(7.0, 4.0, 24.0), "m4_screw")
        self.assertEqual(classify(13.0, 8.0, 28.0), "m8_screw")
        with self.assertRaises(PerceptionError):
            classify(9.5, 6.0, 25.0)

    def test_full_field_scanner_is_count_independent(self):
        image = np.full((1080, 1920, 3), 245, np.uint8)
        _, (left, top, right, bottom) = self.client._object_crop(image)
        y1 = top + (bottom - top) // 3
        y2 = top + 2 * (bottom - top) // 3
        x1 = left + (right - left) // 3
        x2 = left + 2 * (right - left) // 3
        cv2.line(image, (x1 - 24, y1), (x1 + 24, y1 + 6), (15, 15, 15), 5)
        cv2.circle(image, (x1 - 24, y1), 8, (15, 15, 15), -1)
        cv2.line(image, (x2 - 28, y2), (x2 + 28, y2 - 8), (20, 20, 20), 9)
        cv2.circle(image, (x2 - 28, y2), 13, (20, 20, 20), -1)
        proposals = self.client._pixel_screw_proposals(image)
        self.assertGreaterEqual(len(proposals), 2)

    def test_guard_is_a_ceiling_not_expected_count(self):
        self.assertGreaterEqual(MAX_TASK_OBJECTS, 16)
        self.assertEqual(set(CLASS_TO_BIN), {"m4_screw", "m8_screw"})

    def test_residual_scan_excludes_both_destination_bins(self):
        bins = [
            {"color": "red", "bbox": [0.20, 0.20, 0.30, 0.32]},
            {"color": "green", "bbox": [0.60, 0.55, 0.72, 0.70]},
        ]
        objects = [
            {"sort_class": "m4_screw", "grasp_center": [0.25, 0.26]},
            {"sort_class": "m8_screw", "grasp_center": [0.66, 0.62]},
            {"sort_class": "m4_screw", "grasp_center": [0.45, 0.45]},
        ]
        kept, excluded = self.client._exclude_bin_contents(
            objects, bins, (1080, 1920, 3)
        )
        self.assertEqual([item["grasp_center"] for item in kept], [[0.45, 0.45]])
        self.assertEqual(len(excluded), 2)

    def test_relocalization_ignores_large_nearby_seam(self):
        mask = np.zeros((101, 101), np.uint8)
        # This large component is close enough and large enough that the old
        # area-weighted selector preferred it over the screw at the target.
        cv2.rectangle(mask, (0, 1), (37, 99), 255, -1)
        cv2.rectangle(mask, (46, 47), (54, 53), 255, -1)
        target = np.asarray([50.0, 50.0])
        old_component, old_area = self.client._component_near(mask, target)
        self.assertGreater(old_area, 1000)

        original = self.client._fit_screw_shape

        def fake_fit(points, offset, image_shape, expected_class=None):
            if len(points) > 1000:
                raise PerceptionError("unsupported oversized seam")
            return {"sort_class": expected_class, "selected_area": len(points)}

        self.client._fit_screw_shape = fake_fit
        try:
            fit = self.client._fit_expected_screw_component_near(
                mask,
                target,
                np.zeros(2),
                (1080, 1920, 3),
                "m4_screw",
            )
        finally:
            self.client._fit_screw_shape = original
        self.assertLess(fit["selected_area"], 100)


class MotionContractTests(unittest.TestCase):
    def test_calibrated_old_and_new_bin_profiles(self):
        self.assertEqual(classify_bin_profile(85.0)["name"], "old")
        self.assertEqual(classify_bin_profile(135.0)["name"], "new")
        with self.assertRaises(ValueError):
            classify_bin_profile(105.0)

    def test_old_bin_scene_uses_old_projection_and_release(self):
        task = parse_instruction(
            "Sort all M4 screws into the red bin and all M8 screws into the yellow bin."
        )
        client = Sort4CosmosClient("http://127.0.0.1:9")
        targets = client.to_targets(client.mock_scene(task), task)
        self.assertTrue(targets)
        self.assertTrue(all(target.bin_profile == "old" for target in targets))
        self.assertTrue(all(target.bin_body_height_mm == 55.0 for target in targets))
        self.assertTrue(all(target.bin_release_z_mm == -270.0 for target in targets))

    def test_new_bin_height_and_release_clearance(self):
        self.assertEqual(BIN_BODY_HEIGHT_M, 0.060)
        self.assertEqual(BIN_VISUAL_MIDPOINT_M, 0.030)
        self.assertEqual(BIN_RELEASE_Z, -265.0)

    def test_screw_depths_remain_minus_334(self):
        self.assertEqual(GRASP_Z, {"m4_screw": -334.0, "m8_screw": -334.0})

    def test_verified_pick_moves_home_before_relocalization(self):
        target = SortTarget(
            sort_class="m4_screw",
            label="m4",
            object_xy_mm=(360.0, 0.0),
            object_yaw_deg=0.0,
            bin_color="red",
            bin_xy_mm=(420.0, -150.0),
        )
        with tempfile.TemporaryDirectory() as log_dir, patch.dict(
            os.environ,
            {
                "DEMO_LOG_DIR": log_dir,
                "DEMO_ALLOW_REAL_ROBOT": "YES",
                "DEMO_UTOOL1_CONFIRMED": "YES",
                "DEMO_TOUCH_OFF_CONFIRMED": "YES",
            },
        ):
            controller = Sort4Controller()
            controller.phase = "verify_pick"
            controller.task = object()
            controller.last_placed_target = target
            controller.pending_targets = [target]
            controller.vision.target_is_present = lambda image, old: False
            obs = {
                "trial": {"trial_id": "real-camera-clear-test"},
                "images": {"table": np.zeros((32, 32, 3), np.uint8)},
                "tcp_pose": [420.0, -150.0, -225.0, -180.0, 0.0, 0.0],
                "joints": [0.0] * 7,
            }
            rows = controller.act(obs)
        self.assertEqual(controller.phase, "relocalize")
        np.testing.assert_allclose(controller.last_command, SAFE_HOME)
        np.testing.assert_allclose(rows[-1], SAFE_HOME)


if __name__ == "__main__":
    unittest.main()
