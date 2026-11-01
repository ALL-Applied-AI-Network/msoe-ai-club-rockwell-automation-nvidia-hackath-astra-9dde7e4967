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


class Policy10Tests(unittest.TestCase):
    def test_mapping(self):
        task = parse_instruction(config.DEFAULT_INSTRUCTION)
        self.assertEqual(task.class_to_bin, {"round_pill": "red", "long_pill": "green"})

    def test_rejects_other_families(self):
        for word in ("screws", "batteries", "cubes"):
            with self.assertRaises(ValueError):
                parse_instruction(config.DEFAULT_INSTRUCTION + " and " + word)

    def test_requires_both_destinations(self):
        with self.assertRaises(ValueError):
            parse_instruction("Sort the pills by size")

    def test_robot_prompt_can_swap_destinations(self):
        task = parse_instruction(
            "Sort all round pills into the green bin and all long pills into the red bin."
        )
        self.assertEqual(task.class_to_bin, {"round_pill": "green", "long_pill": "red"})

    def test_rejects_same_destination_for_both_classes(self):
        with self.assertRaisesRegex(ValueError, "different supported bins"):
            parse_instruction(
                "Sort all round pills into the red bin and all long pills into the red bin."
            )

    def test_old_red_yellow_prompt_is_supported(self):
        task = parse_instruction(
            "Sort all round pills into the red bin and all long pills into the yellow bin."
        )
        self.assertEqual(task.class_to_bin, {"round_pill": "red", "long_pill": "yellow"})

    def test_calibrated_old_and_new_bin_profiles(self):
        self.assertEqual(classify_bin_profile(85.0)["name"], "old")
        self.assertEqual(classify_bin_profile(135.0)["name"], "new")
        with self.assertRaises(ValueError):
            classify_bin_profile(105.0)

    def test_old_bin_scene_uses_old_projection_and_release(self):
        task = parse_instruction(
            "Sort all round pills into the red bin and all long pills into the yellow bin."
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

    def test_metric_pill_classes(self):
        self.assertEqual(TrainedDetectorClient._classify_geometry(11.5, 11.5), "round_pill")
        self.assertEqual(TrainedDetectorClient._classify_geometry(18.7, 8.5), "long_pill")
        with self.assertRaises(PerceptionError):
            TrainedDetectorClient._classify_geometry(30.0, 20.0)

    def test_task_specific_model_allowlist(self):
        self.assertEqual(set(config.MODEL_CLASSES), {"round_pill", "long_pill"})

    def test_count_is_a_ceiling_not_expected_inventory(self):
        self.assertEqual(config.MAX_TASK_OBJECTS, 24)
        task = parse_instruction(config.DEFAULT_INSTRUCTION)
        scene = TrainedDetectorClient.mock_scene(task)
        self.assertEqual(len(scene["objects"]), 4)
        self.assertNotIn("expected_counts", inspect.getsource(config))

    def test_depth_contract(self):
        self.assertEqual(GRASP_Z, {"round_pill": -334.0, "long_pill": -334.0})
        self.assertEqual(config.BIN_BODY_HEIGHT_M, 0.060)
        self.assertEqual(config.BIN_VISUAL_MIDPOINT_M, 0.030)
        self.assertEqual(BIN_RELEASE_Z, -265.0)

    def test_residual_scan_is_dynamic(self):
        source = inspect.getsource(TrainedDetectorClient.residual_targets)
        self.assertIn("detect_objects", source)
        self.assertNotIn("initial_inventory", source)

    def test_local_scan_proposes_round_and_long_coloured_pills(self):
        client = TrainedDetectorClient()
        image = np.full((1080, 1920, 3), 245, dtype=np.uint8)
        _, (left, top, right, bottom) = client._object_crop(image)
        round_center = (left + (right - left) // 3, top + (bottom - top) // 2)
        long_center = (left + 2 * (right - left) // 3, top + (bottom - top) // 2)
        cv2.circle(image, round_center, 14, (240, 210, 25), -1)
        cv2.ellipse(image, long_center, (28, 10), 35, 0, 360, (220, 30, 30), -1)

        proposals = client._local_pill_proposals(image)

        self.assertEqual(len(proposals), 2)
        self.assertEqual(
            {item["sort_class"] for item in proposals}, {"round_pill", "long_pill"}
        )
        self.assertTrue(
            all(item["source"] == "local chromatic component" for item in proposals)
        )

    def test_metric_geometry_overrides_wrong_proposal_label(self):
        client = TrainedDetectorClient()
        image = np.full((200, 200, 3), 245, dtype=np.uint8)
        cv2.ellipse(image, (100, 100), (40, 15), 25, 0, 360, (220, 30, 30), -1)
        proposal = {
            "sort_class": "round_pill",
            "center": [0.5, 0.5],
            "bbox": [0.25, 0.35, 0.75, 0.65],
        }
        with patch.object(client, "_metric_distance_mm", side_effect=[18.7, 8.5]):
            fit = client._fit_once(image, proposal, 16.0)
        self.assertEqual(fit["sort_class"], "long_pill")

    def test_bin_detection_rejects_yellow_caution_tape_outside_workspace(self):
        client = TrainedDetectorClient()
        image = np.full((1080, 1920, 3), 245, dtype=np.uint8)
        _, (left, top, right, bottom) = client._object_crop(image)
        y = top + (bottom - top) // 2
        cv2.rectangle(image, (left + 55, y - 45), (left + 145, y + 45), (25, 180, 45), -1)
        cv2.rectangle(image, (right - 145, y - 45), (right - 55, y + 45), (220, 25, 25), -1)
        cv2.rectangle(image, (1700, 180), (1880, 800), (245, 215, 20), -1)

        bins = {item["color"]: item for item in client._detect_bins(image)}

        self.assertGreater(bins["green"]["center"][0], left / image.shape[1])
        self.assertLess(bins["green"]["center"][0], right / image.shape[1])
        self.assertEqual(len(bins["green"]["bbox"]), 4)

    def test_round_pill_yaw_is_canonical_zero(self):
        task = parse_instruction(config.DEFAULT_INSTRUCTION)
        client = TrainedDetectorClient()
        scene = client.mock_scene(task)
        scene["objects"][0]["axis_endpoints"] = [[0.39, 0.58], [0.39, 0.66]]
        targets = client.to_targets(scene, task)
        round_targets = [item for item in targets if item.sort_class == "round_pill"]
        self.assertTrue(round_targets)
        self.assertTrue(all(item.object_yaw_deg == 0.0 for item in round_targets))


if __name__ == "__main__":
    unittest.main()
