import inspect
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from battery_policy.perception import TrainedDetectorClient as BatteryClient
from family_policy import config
from family_policy.bin_profiles import classify_bin_profile
from family_policy.controller import SAFE_HOME, VariableSortController
from family_policy.instruction import parse_instruction
from family_policy.perception import (
    MultiFamilyDetectorClient,
    PerceptionError,
)
from pill_policy.perception import TrainedDetectorClient as PillClient
from sort4_policy.perception import Sort4CosmosClient as ScrewClient


FAMILY_PROMPT = (
    "Sort all screws into the red bin, all batteries into the green bin, "
    "and all pills into the yellow bin."
)
SUBTYPE_PROMPT = (
    "Sort all M4 screws into the blue bin, all M8 screws into the green bin, "
    "all round pills into the yellow bin, and all long pills into the red bin."
)


class InstructionTests(unittest.TestCase):
    def test_family_level_prompt_expands_to_all_six_classes(self):
        task = parse_instruction(FAMILY_PROMPT)
        self.assertEqual(
            task.class_to_bin,
            {
                "m4_screw": "red",
                "m8_screw": "red",
                "aaa_battery": "green",
                "aa_battery": "green",
                "round_pill": "yellow",
                "long_pill": "yellow",
            },
        )

    def test_user_sample_typoes_are_unambiguous_aliases(self):
        task = parse_instruction(
            "Sort all srews in Red bin, sort all batteris in grean bin "
            "and sort all pill in yellow bin."
        )
        self.assertEqual(task.class_to_bin, parse_instruction(FAMILY_PROMPT).class_to_bin)

    def test_subtype_prompt_selects_only_four_classes(self):
        task = parse_instruction(
            "sort all m4 screws in blue bin, sort all m8 screws in green bin, "
            "sort all roung pill in yello bin, sort all long pill into red bin."
        )
        self.assertEqual(
            task.class_to_bin,
            {
                "m4_screw": "blue",
                "m8_screw": "green",
                "round_pill": "yellow",
                "long_pill": "red",
            },
        )

    def test_family_and_subtype_conflict_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "mixes family and subtype"):
            parse_instruction(
                "Sort all screws into the red bin and M8 screws into the green bin."
            )

    def test_missing_destination_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing a destination"):
            parse_instruction("Sort all M4 screws and all M8 screws into the green bin.")

    def test_unsupported_family_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "only supports"):
            parse_instruction("Sort all cubes into the red bin.")


class CompositePerceptionTests(unittest.TestCase):
    def test_family_prompt_mock_contains_all_six_classes(self):
        task = parse_instruction(FAMILY_PROMPT)
        client = MultiFamilyDetectorClient()
        targets = client.to_targets(client.mock_scene(task), task)
        self.assertEqual({target.sort_class for target in targets}, set(config.MODEL_CLASSES))
        self.assertEqual(len(targets), 6)
        self.assertTrue(all(target.bin_profile == "old" for target in targets))
        self.assertTrue(all(target.bin_release_z_mm == -270.0 for target in targets))

    def test_subtype_prompt_does_not_activate_battery_detector(self):
        task = parse_instruction(SUBTYPE_PROMPT)
        client = MultiFamilyDetectorClient()
        targets = client.to_targets(client.mock_scene(task), task)
        self.assertEqual(
            {target.sort_class for target in targets},
            {"m4_screw", "m8_screw", "round_pill", "long_pill"},
        )
        self.assertNotIn("battery", client.family_tasks)

    def test_new_red_green_bins_keep_new_release(self):
        task = parse_instruction(
            "Sort all M4 screws into the red bin and all M8 screws into the green bin."
        )
        client = MultiFamilyDetectorClient()
        targets = client.to_targets(client.mock_scene(task), task)
        self.assertTrue(all(target.bin_profile == "new" for target in targets))
        self.assertTrue(all(target.bin_release_z_mm == -265.0 for target in targets))

    def test_old_and_new_profile_classifier_is_fail_closed(self):
        self.assertEqual(classify_bin_profile(85.0)["name"], "old")
        self.assertEqual(classify_bin_profile(135.0)["name"], "new")
        with self.assertRaises(ValueError):
            classify_bin_profile(105.0)

    def test_blue_bin_mask_is_supported_by_every_family(self):
        hsv = np.asarray([[[110, 200, 200]]], np.uint8)
        for client in (ScrewClient(), BatteryClient(), PillClient()):
            self.assertEqual(int(client._color_mask(hsv, "blue")[0, 0]), 255)

    def test_cross_family_overlap_fails_closed(self):
        task = parse_instruction(FAMILY_PROMPT)
        client = MultiFamilyDetectorClient()
        targets = client.to_targets(client.mock_scene(task), task)
        overlapped = [targets[0], replace(targets[2], object_xy_mm=targets[0].object_xy_mm)]
        with self.assertRaisesRegex(PerceptionError, "cross-family ambiguous"):
            client._sort_and_validate(overlapped, task, allow_empty=False)

    def test_inventory_limit_is_a_ceiling_not_a_fixed_count(self):
        self.assertEqual(config.MAX_TASK_OBJECTS, 48)
        self.assertNotIn("expected_counts", inspect.getsource(config))

    def test_shared_classifier_is_filtered_per_family(self):
        self.assertIn("allowed_classes", inspect.getsource(BatteryClient._classifier_proposals))
        self.assertIn("allowed_classes", inspect.getsource(PillClient._detector_proposals))
        service = (Path(__file__).resolve().parents[1] / "detector_service.py").read_text()
        self.assertIn('request.get("allowed_classes")', service)


class ControllerTests(unittest.TestCase):
    def test_verified_pick_moves_home_before_relocalization(self):
        task = parse_instruction(FAMILY_PROMPT)
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
                "trial": {"trial_id": "real-camera-clear-test", "instruction": FAMILY_PROMPT},
                "images": {"table": np.zeros((32, 32, 3), np.uint8)},
                "tcp_pose": [420.0, -150.0, -225.0, -180.0, 0.0, 0.0],
                "joints": [0.0] * 7,
            }
            rows = controller.act(obs)
        self.assertEqual(controller.phase, "relocalize")
        np.testing.assert_allclose(controller.last_command, SAFE_HOME)
        np.testing.assert_allclose(rows[-1], SAFE_HOME)

    def test_controller_uses_prompt_selected_grasp_depths(self):
        self.assertEqual(set(config.GRASP_Z_MM), set(config.MODEL_CLASSES))
        self.assertTrue(all(value == -334.0 for value in config.GRASP_Z_MM.values()))


if __name__ == "__main__":
    unittest.main()
