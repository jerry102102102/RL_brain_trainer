import unittest
from pathlib import Path

import yaml

from hrl_trainer.v5.rollout_integrity import (
    ROLLOUT_ARTIFACT_SCHEMA_VERSION,
    compare_replay_determinism,
    rollout_digest_sha256,
    validate_rollout_payload,
)
from hrl_trainer.v5.rl_action import (
    RLActionValidationError,
    action_to_skill_command,
    validate_rl_action_v1,
    validate_skill_command_boundary,
)
from hrl_trainer.v5.rl_observation import (
    ObjectPoseEstimate,
    Pose6D,
    RLObservationValidationError,
    RobotState,
    build_rl_observation_v1,
    validate_rl_observation_v1,
)
from hrl_trainer.v5.reward_composer import RewardComposer, RewardTermInput, RewardTermWeights
from hrl_trainer.v5.workspace_zone_map import (
    DEFAULT_WORKSPACE_ZONE_MAP_PATH,
    WorkspaceZoneMap,
    load_runtime_workspace_zone_map,
)


class TestV5Wp15RlObservation(unittest.TestCase):
    def test_observation_validator_accepts_valid_payload(self):
        observation = build_rl_observation_v1(
            obs_latent=[0.1, 0.2, 0.3, 0.4],
            robot_state=RobotState(
                joint_positions=(0.0, 0.1, 0.2),
                joint_velocities=(0.0, 0.0, 0.0),
                ee_pose=Pose6D(xyz=(0.1, -0.2, 1.1), rpy=(3.14, 0.0, 0.1)),
                gripper_opening=0.6,
            ),
            stage_flag="APPROACH",
            target_slot="shelf_B1",
            target_zone="shelf_right",
            source_slot="shelf_A1",
            active_zone="transfer_corridor",
            object_pose_est=ObjectPoseEstimate(
                object_id="tray1",
                xyz=(0.1, -1.0, 1.2),
                rpy=(3.14, 0.0, 0.0),
                confidence=0.95,
                pos_std=0.01,
                yaw_std=0.03,
                stamp_sec=12.3,
            ),
        )
        validate_rl_observation_v1(observation)

    def test_observation_validator_rejects_hidden_gt_fields(self):
        payload = {
            "schema_version": "v1",
            "obs_latent": [0.1, 0.2],
            "robot_state": {
                "joint_positions": [0.0, 0.0],
                "joint_velocities": [0.0, 0.0],
                "ee_pose": {"xyz": [0.0, 0.0, 1.0], "rpy": [0.0, 0.0, 0.0]},
                "gripper_opening": 0.5,
            },
            "stage_flag": "TRANSFER",
            "target_slot": "shelf_B1",
            "target_zone": "shelf_right",
            "gt_object_pose": {"xyz": [0.0, 0.0, 0.0]},
        }
        with self.assertRaises(RLObservationValidationError):
            validate_rl_observation_v1(payload)


class TestV5Wp15RlAction(unittest.TestCase):
    def test_action_validator_and_adapter(self):
        action = {
            "schema_version": "v1",
            "skill_mode": "APPROACH",
            "delta_pose": {"xyz": [0.02, -0.01, 0.03], "rpy": [0.0, 0.0, 0.1]},
            "gripper_cmd": "HOLD",
            "speed_profile_id": "SLOW",
            "guard": {"keep_level": True, "max_tilt": 0.4, "min_clearance": 0.03},
        }
        validate_rl_action_v1(action)
        command = action_to_skill_command(action)
        validate_skill_command_boundary(command)
        self.assertEqual(command.skill_mode, "APPROACH")
        self.assertIsNotNone(command.delta_pose)
        self.assertIsNone(command.ee_target_pose)

    def test_action_validator_rejects_l3_fields(self):
        bad_action = {
            "schema_version": "v1",
            "skill_mode": "PLACE",
            "ee_target_pose": {"xyz": [0.0, 0.0, 1.2], "rpy": [3.14, 0.0, 0.0]},
            "gripper_cmd": "OPEN",
            "speed_profile_id": "NORMAL",
            "guard": {"keep_level": True, "max_tilt": 0.6, "min_clearance": 0.02},
            "joint_trajectory": [],
        }
        with self.assertRaises(RLActionValidationError):
            validate_rl_action_v1(bad_action)


class TestV5Wp15RewardComposer(unittest.TestCase):
    def test_reward_composer_returns_step_and_episode_breakdown(self):
        composer = RewardComposer(weights=RewardTermWeights(progress=1.0, safety=2.0, smoothness=0.5, coverage=0.1, subgoal=3.0))
        step0 = composer.compose_step(
            0,
            RewardTermInput(progress=0.3, safety=-0.2, smoothness=-0.1, coverage=0.6, subgoal=0.0),
        )
        step1 = composer.compose_step(
            1,
            RewardTermInput(progress=0.4, safety=-0.1, smoothness=-0.1, coverage=0.2, subgoal=1.0),
            terminal=True,
            notes=["subgoal_complete"],
        )
        episode = composer.compose_episode([step0, step1], terminal_reason="SUCCESS")
        self.assertAlmostEqual(step0.total_reward, -0.09, places=6)
        self.assertTrue(step1.terminal)
        self.assertEqual(episode.terminal_reason, "SUCCESS")
        self.assertAlmostEqual(episode.total_reward, step0.total_reward + step1.total_reward, places=6)


class TestV5Wp15WorkspaceZoneMap(unittest.TestCase):
    def test_runtime_workspace_zone_map_loads_default_config(self):
        self.assertTrue(Path(DEFAULT_WORKSPACE_ZONE_MAP_PATH).exists())
        zone_map = load_runtime_workspace_zone_map()
        self.assertGreaterEqual(len(zone_map.zones), 3)
        self.assertGreaterEqual(len(zone_map.anchors), 3)
        self.assertEqual(zone_map.zone_for_point((0.90, -1.16, 1.22)), "shelf_left")

    def test_zone_map_rejects_anchor_zone_mismatch(self):
        payload = {
            "zones": [
                {
                    "zone_id": "zone1",
                    "region_world": {"center_xyz": [0.0, 0.0, 0.0], "size_xyz": [1.0, 1.0, 1.0], "yaw": 0.0},
                    "hover_anchor_ids": ["a1"],
                }
            ],
            "anchors": [
                {
                    "anchor_id": "a1",
                    "zone_id": "zone2",
                    "pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                }
            ],
        }
        with self.assertRaises(ValueError):
            WorkspaceZoneMap.from_dict(payload)


class TestV5Wp15RolloutIntegrity(unittest.TestCase):
    def _sample_rollout(self) -> dict:
        return {
            "schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
            "episode_id": "ep-0001",
            "curriculum_level": "easy",
            "seed": 101,
            "steps": [
                {"t": 0, "action": {"skill_mode": "APPROACH"}, "reward_total": 0.2},
                {"t": 1, "action": {"skill_mode": "PLACE"}, "reward_total": 1.0},
            ],
        }

    def test_rollout_validator_accepts_minimal_payload(self):
        payload = self._sample_rollout()
        self.assertEqual(validate_rollout_payload(payload), [])

    def test_rollout_digest_is_key_order_invariant(self):
        payload_a = self._sample_rollout()
        payload_b = {
            "steps": payload_a["steps"],
            "seed": payload_a["seed"],
            "curriculum_level": payload_a["curriculum_level"],
            "episode_id": payload_a["episode_id"],
            "schema_version": payload_a["schema_version"],
        }
        self.assertEqual(rollout_digest_sha256(payload_a), rollout_digest_sha256(payload_b))

    def test_compare_replay_determinism_detects_drift(self):
        reference = self._sample_rollout()
        replay = self._sample_rollout()
        replay["steps"][1]["reward_total"] = 0.9
        out = compare_replay_determinism(reference, replay)
        self.assertFalse(out["match"])
        self.assertIn("mismatch_reason", out)


class TestV5Wp15CurriculumFixtures(unittest.TestCase):
    def test_curriculum_fixture_levels_and_seed_sets(self):
        cfg_dir = Path(__file__).resolve().parents[1] / "config"
        fixture_names = [
            ("easy", "v5_curriculum_easy.yaml"),
            ("medium", "v5_curriculum_medium.yaml"),
            ("hard", "v5_curriculum_hard.yaml"),
        ]
        all_seeds: list[int] = []
        for expected_level, filename in fixture_names:
            with (cfg_dir / filename).open("r", encoding="utf-8") as fp:
                payload = yaml.safe_load(fp) or {}
            self.assertEqual(payload["schema_version"], "v5_curriculum.v1")
            self.assertEqual(payload["level"], expected_level)
            self.assertTrue(isinstance(payload["seed_set"], list) and len(payload["seed_set"]) >= 3)
            all_seeds.extend(payload["seed_set"])
        self.assertEqual(len(all_seeds), len(set(all_seeds)))


if __name__ == "__main__":
    unittest.main()
