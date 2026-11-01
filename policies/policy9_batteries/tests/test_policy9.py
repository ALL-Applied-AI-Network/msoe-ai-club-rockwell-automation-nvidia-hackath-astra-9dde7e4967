import inspect
import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from family_policy import config
from family_policy.bin_profiles import classify_bin_profile
from family_policy.controller import (
    BIN_RELEASE_Z,
    GRASP_Z,
    SAFE_HOME,
    VariableSortController,
)
from family_policy.instruction import parse_instruction
from family_policy.perception import PerceptionError, TrainedDetectorClient


class Policy9Tests(unittest.TestCase):
    def test_mapping(self):
        task = parse_instruction(config.DEFAULT_INSTRUCTION)
        self.assertEqual(task.class_to_bin, {"aaa_battery": "red", "aa_battery": "green"})

    def test_rejects_other_families(self):
        for word in ("screws", "pills", "cubes"):
            with self.assertRaises(ValueError):
                parse_instruction(config.DEFAULT_INSTRUCTION + " and " + word)

    def test_requires_both_destinations(self):
        with self.assertRaises(ValueError):
            parse_instruction("Put every battery in the red bin")

    def test_robot_prompt_can_swap_destinations(self):
        task = parse_instruction(
            "Sort all AAA batteries into the green bin and all AA batteries into the red bin."
        )
        self.assertEqual(task.class_to_bin, {"aaa_battery": "green", "aa_battery": "red"})

    def test_rejects_same_destination_for_both_classes(self):
        with self.assertRaisesRegex(ValueError, "different supported bins"):
            parse_instruction(
                "Sort all AAA batteries into the red bin and all AA batteries into the red bin."
            )

    def test_old_red_yellow_prompt_is_supported(self):
        task = parse_instruction(
            "Sort all AAA batteries into the red bin and all AA batteries into the yellow bin."
        )
        self.assertEqual(task.class_to_bin, {"aaa_battery": "red", "aa_battery": "yellow"})

    def test_calibrated_old_and_new_bin_profiles(self):
        self.assertEqual(classify_bin_profile(85.0)["name"], "old")
        self.assertEqual(classify_bin_profile(135.0)["name"], "new")
        with self.assertRaises(ValueError):
            classify_bin_profile(105.0)

    def test_old_bin_scene_uses_old_projection_and_release(self):
        task = parse_instruction(
            "Sort all AAA batteries into the red bin and all AA batteries into the yellow bin."
        )
        client = TrainedDetectorClient()
        targets = client.to_targets(client.mock_scene(task), task)
        self.assertTrue(targets)
        self.assertTrue(all(target.bin_profile == "old" for target in targets))
        self.assertTrue(all(target.bin_body_height_mm == 55.0 for target in targets))
        self.assertTrue(all(target.bin_release_z_mm == -270.0 for target in targets))

    def test_verified_pick_moves_home_before_relocalization(self):
        task = parse_instruction(config.DEFAULT_INSTRUCTION)
        with tempfile.TemporaryDirectory() as log_dir, patch.dict(
            os.environ,
            {
                "DEMO_LOG_DIR": log_dir,
                "DEMO_ALLOW_REAL_ROBOT": "YES",
                "DEMO_UTOOL1_CONFIRMED": "YES",
                "DEMO_TOUCH_OFF_CONFIRMED": "YES",
            },
        ):
            controller = VariableSortController()
            target = controller.vision.to_targets(
                controller.vision.mock_scene(task), task
            )[0]
            controller.phase = "verify_pick"
            controller.task = task
            controller.last_placed_target = target
            controller.pending_targets = [target]
            controller.vision.target_is_present = lambda image, old: False
            obs = {
                "trial": {"trial_id": "real-camera-clear-test", "instruction": config.DEFAULT_INSTRUCTION},
                "images": {"table": np.zeros((32, 32, 3), np.uint8)},
                "tcp_pose": [420.0, -150.0, -225.0, -180.0, 0.0, 0.0],
                "joints": [0.0] * 7,
            }
            rows = controller.act(obs)
        self.assertEqual(controller.phase, "relocalize")
        np.testing.assert_allclose(controller.last_command, SAFE_HOME)
        np.testing.assert_allclose(rows[-1], SAFE_HOME)

    def test_metric_battery_classes(self):
        classify = TrainedDetectorClient._classify_geometry
        for sort_class, measurements in {
            "aaa_battery": ((44.0, 10.2), (46.5, 15.1), (47.6, 13.8)),
            "aa_battery": ((49.0, 14.0), (56.3, 19.5)),
        }.items():
            for length, width in measurements:
                self.assertEqual(classify(length, width), sort_class)
        with self.assertRaises(PerceptionError):
            classify(48.95, 15.0)
        self.assertTrue(config.VERIFY_CLASS_BY_GEOMETRY)

    def test_geometry_can_override_wrong_model_colour_class(self):
        # The first real CSI frame made the trained model call both cells AAA.
        # Geometry must independently recover the larger AA class.
        self.assertEqual(
            TrainedDetectorClient._classify_geometry(56.3, 19.5),
            "aa_battery",
        )

    def test_task_specific_model_allowlist(self):
        self.assertEqual(set(config.MODEL_CLASSES), {"aa_battery", "aaa_battery"})

    def test_count_is_a_ceiling_not_expected_inventory(self):
        self.assertEqual(config.MAX_TASK_OBJECTS, 24)
        task = parse_instruction(config.DEFAULT_INSTRUCTION)
        scene = TrainedDetectorClient.mock_scene(task)
        self.assertEqual(len(scene["objects"]), 4)
        self.assertNotIn("expected_counts", inspect.getsource(config))

    def test_depth_contract(self):
        self.assertEqual(GRASP_Z, {"aaa_battery": -334.0, "aa_battery": -334.0})
        self.assertEqual(config.BIN_BODY_HEIGHT_M, 0.060)
        self.assertEqual(config.BIN_VISUAL_MIDPOINT_M, 0.030)
        self.assertEqual(BIN_RELEASE_Z, -265.0)

    def test_residual_scan_is_dynamic(self):
        source = inspect.getsource(TrainedDetectorClient.residual_targets)
        self.assertIn("detect_objects", source)
        self.assertNotIn("initial_inventory", source)

    def test_cosmos_grid_bbox_is_normalized_from_axis(self):
        bbox = TrainedDetectorClient._cosmos_bbox(
            {
                "bbox": [446, 649, 460, 698],
                "axis_endpoints": [[0.447, 0.650], [0.457, 0.694]],
            },
            (1080, 1920, 3),
        )
        self.assertIsNotNone(bbox)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in bbox))
        self.assertLess(bbox[0], 0.447)
        self.assertGreater(bbox[2], 0.457)

    def test_relocalization_uses_local_window_not_second_cosmos_call(self):
        source = inspect.getsource(TrainedDetectorClient.relocalize_targets)
        self.assertIn("_target_proposal", source)
        self.assertNotIn("detect_objects", source)

    def test_bin_detection_rejects_yellow_caution_tape_outside_workspace(self):
        client = TrainedDetectorClient()
        image = np.full((1080, 1920, 3), 245, dtype=np.uint8)
        _, (left, top, right, bottom) = client._object_crop(image)
        y = top + (bottom - top) // 2
        cv2.rectangle(image, (left + 55, y - 45), (left + 145, y + 45), (25, 180, 45), -1)
        cv2.rectangle(image, (right - 145, y - 45), (right - 55, y + 45), (220, 25, 25), -1)
        cv2.rectangle(image, (1700, 180), (1880, 800), (245, 215, 20), -1)

        bins = {item["color"]: item for item in client._detect_bins(image)}

        self.assertLess(bins["green"]["center"][0], right / image.shape[1])
        self.assertGreater(bins["green"]["center"][0], left / image.shape[1])
        self.assertEqual(len(bins["green"]["bbox"]), 4)
        self.assertEqual(len(bins["red"]["bbox"]), 4)


if __name__ == "__main__":
    unittest.main()
