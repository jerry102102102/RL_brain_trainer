from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.kinematic_phase1.envs.reward_approach import ApproachRewardConfig, compute_approach_reward


class TestKinematicPhase1ApproachReward(unittest.TestCase):
    def test_near_field_orientation_progress_is_stronger_inside_pre_near_goal(self) -> None:
        goal_pose = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        prev_pose = np.array([0.0, 0.0, 0.0, 0.30, 0.0, 0.0], dtype=float)
        curr_pose = np.array([0.0, 0.0, 0.0, 0.20, 0.0, 0.0], dtype=float)
        _, far_components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=False,
            prev_in_near_goal=False,
            curr_in_near_goal=False,
            dwell_count=0,
            joint_limit_margin_min=1.0,
            success=False,
            config=ApproachRewardConfig(orientation_progress_weight=1.0, near_field_orientation_progress_weight=2.0),
        )
        _, near_components = compute_approach_reward(
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
            config=ApproachRewardConfig(orientation_progress_weight=1.0, near_field_orientation_progress_weight=2.0),
        )
        self.assertGreater(near_components["orientation_progress"], far_components["orientation_progress"])
        self.assertGreater(near_components["near_field_orientation_progress"], 0.0)

    def test_coarse_orientation_bonus_requires_position_and_orientation(self) -> None:
        goal_pose = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        prev_pose = np.array([0.0, 0.0, 0.0, 0.40, 0.0, 0.0], dtype=float)
        curr_pose_good = np.array([0.0, 0.0, 0.0, 0.10, 0.0, 0.0], dtype=float)
        curr_pose_bad = np.array([0.0, 0.0, 0.0, 0.60, 0.0, 0.0], dtype=float)
        _, good_components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose_good,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=False,
            curr_in_near_goal=False,
            dwell_count=0,
            joint_limit_margin_min=1.0,
            success=False,
            config=ApproachRewardConfig(coarse_orientation_bonus=0.05, coarse_orientation_bonus_threshold_rad=0.35),
        )
        _, bad_components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose_bad,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=False,
            curr_in_near_goal=False,
            dwell_count=0,
            joint_limit_margin_min=1.0,
            success=False,
            config=ApproachRewardConfig(coarse_orientation_bonus=0.05, coarse_orientation_bonus_threshold_rad=0.35),
        )
        self.assertGreater(good_components["coarse_orientation_bonus"], 0.0)
        self.assertEqual(bad_components["coarse_orientation_bonus"], 0.0)

    def test_near_goal_reentry_bonus_decays(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        prev_pose = np.array([0.04, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        curr_pose = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        cfg = ApproachRewardConfig(near_goal_bonus=1.0, near_goal_bonus_decay=0.5)
        _, first_components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=False,
            curr_in_near_goal=True,
            dwell_count=1,
            near_goal_entry_count=1,
            near_goal_drift_count=0,
            joint_limit_margin_min=1.0,
            success=False,
            config=cfg,
        )
        _, second_components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=False,
            curr_in_near_goal=True,
            dwell_count=1,
            near_goal_entry_count=2,
            near_goal_drift_count=0,
            joint_limit_margin_min=1.0,
            success=False,
            config=cfg,
        )
        self.assertEqual(first_components["near_goal_bonus"], 1.0)
        self.assertEqual(second_components["near_goal_bonus"], 0.5)

    def test_drift_penalty_escalates_after_repeated_near_goal_drift(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        prev_pose = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        curr_pose = np.array([0.025, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        cfg = ApproachRewardConfig(
            drift_penalty_weight=2.0,
            drift_penalty_escalation_start=2,
            drift_penalty_escalation_per_count=0.5,
        )
        _, early_components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=2,
            near_goal_entry_count=1,
            near_goal_drift_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=cfg,
        )
        _, late_components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=2,
            near_goal_entry_count=1,
            near_goal_drift_count=4,
            joint_limit_margin_min=1.0,
            success=False,
            config=cfg,
        )
        self.assertLess(late_components["drift_penalty"], early_components["drift_penalty"])
        self.assertGreater(late_components["drift_penalty_scale"], early_components["drift_penalty_scale"])

    def test_near_goal_leave_penalty_triggers_when_exiting_zone(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        prev_pose = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        curr_pose = np.array([0.04, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        _, components = compute_approach_reward(
            prev_pose6=prev_pose,
            curr_pose6=curr_pose,
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            curr_in_pre_near_goal=True,
            prev_in_near_goal=True,
            curr_in_near_goal=False,
            dwell_count=0,
            near_goal_entry_count=1,
            near_goal_drift_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=ApproachRewardConfig(near_goal_leave_penalty=0.35),
        )
        self.assertEqual(components["near_goal_leave_penalty"], -0.35)
