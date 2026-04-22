from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.kinematic_phase1.envs.arm_kinematic_env import ArmKinematicEnv
from hrl_trainer.kinematic_phase1.kinematics.fk_interface import compute_ee_pose6


class TestKinematicPhase1Env(unittest.TestCase):
    def test_reset_returns_expected_observation_keys(self) -> None:
        env = ArmKinematicEnv()
        obs, info = env.reset(seed=123)

        self.assertEqual(
            set(obs.keys()),
            {
                "q",
                "dq",
                "prev_action",
                "goal_pos_err",
                "goal_ori_err",
                "wp_pos_err",
                "wp_ori_err",
                "next_wp_pos_err",
                "next_wp_ori_err",
                "task_type",
                "mode_flag",
                "progress",
                "joint_limit_margin",
            },
        )
        self.assertEqual(obs["q"].shape, (7,))
        self.assertEqual(obs["goal_pos_err"].shape, (3,))
        self.assertEqual(obs["task_type"].tolist(), [1.0, 0.0, 0.0])
        self.assertIn("goal_pose6", info)

    def test_step_returns_reward_breakdown_and_clipped_state(self) -> None:
        env = ArmKinematicEnv()
        env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step(np.ones(7, dtype=np.float32) * 2.0)

        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("reward_components", info)
        self.assertLessEqual(float(np.max(np.abs(obs["q"]))), 1.0)
        self.assertGreaterEqual(float(np.min(obs["joint_limit_margin"])), 0.0)

    def test_success_termination_for_exact_goal_reset(self) -> None:
        env = ArmKinematicEnv()
        _, reset_info = env.reset(seed=9)
        env.reset(options={"initial_q": reset_info["goal_q"], "goal_q": reset_info["goal_q"], "goal_pose6": reset_info["goal_pose6"]})
        terminated = False
        info = {}
        for _ in range(env.config.termination_config.success_dwell_steps):
            _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertTrue(info["success"])

    def test_curriculum_stage_reset_uses_stage_goal(self) -> None:
        env = ArmKinematicEnv()
        env.set_curriculum_stage(3)
        _, info = env.reset(seed=123)

        self.assertEqual(info["curriculum_stage_index"], 3)
        self.assertEqual(info["curriculum_stage_name"], "region_large")
        expected_goal_pose6 = compute_ee_pose6(np.asarray(info["goal_q"], dtype=float))
        np.testing.assert_allclose(info["goal_pose6"], expected_goal_pose6, atol=1e-8)
