from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.kinematic_phase1.envs.reward_fn import RewardConfig, compute_reward


class TestKinematicPhase1Reward(unittest.TestCase):
    def test_reward_prefers_progress_toward_goal(self) -> None:
        prev_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        goal_pose = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        closer_pose = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        farther_pose = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        toward_reward, toward_components = compute_reward(
            prev_pose6=prev_pose,
            curr_pose6=closer_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=False,
            curr_in_near_goal=True,
            dwell_count=0,
            joint_limit_margin_min=1.0,
            success=False,
        )
        away_reward, away_components = compute_reward(
            prev_pose6=closer_pose,
            curr_pose6=farther_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=False,
            prev_in_near_goal=True,
            curr_in_near_goal=False,
            dwell_count=0,
            joint_limit_margin_min=1.0,
            success=False,
        )
        self.assertGreater(toward_components["position_progress"], 0.0)
        self.assertLess(away_components["position_progress"], 0.0)
        self.assertGreater(toward_reward, away_reward)
        self.assertGreater(toward_components["near_goal_bonus"], 0.0)
        self.assertLessEqual(away_components["drift_penalty"], 0.0)

    def test_reward_gives_dwell_bonus_after_staying_in_near_goal(self) -> None:
        goal_pose = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        prev_pose = np.array([0.081, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        curr_pose = np.array([0.082, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        reward, components = compute_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=2,
            joint_limit_margin_min=1.0,
            success=False,
            config=RewardConfig(near_goal_bonus=0.1, dwell_bonus=0.12),
        )
        self.assertAlmostEqual(components["near_goal_bonus"], 0.0)
        self.assertGreater(components["dwell_bonus"], 0.0)
        self.assertIsInstance(reward, float)

    def test_reward_gives_pre_near_goal_bonus_before_true_near_goal(self) -> None:
        goal_pose = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        prev_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        curr_pose = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        _, components = compute_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=False,
            curr_in_near_goal=False,
            dwell_count=0,
            joint_limit_margin_min=1.0,
            success=False,
            config=RewardConfig(pre_near_goal_bonus=0.03),
        )
        self.assertGreater(components["pre_near_goal_bonus"], 0.0)
