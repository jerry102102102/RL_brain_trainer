from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.kinematic_phase1.envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from hrl_trainer.kinematic_phase1.envs.reset_samplers import DockResetConfig
from hrl_trainer.kinematic_phase1.envs.reward_dock import DockRewardConfig, compute_dock_reward
from hrl_trainer.kinematic_phase1.envs.switching_wrapper import SwitchingConfig, TwoPolicySwitcher
from hrl_trainer.kinematic_phase1.envs.termination import TerminationConfig, evaluate_termination


class TestKinematicPhase1Split(unittest.TestCase):
    def test_approach_and_dock_mode_flags_are_distinct(self) -> None:
        approach_env = ArmKinematicEnv()
        approach_obs, _ = approach_env.reset(seed=11, options={"policy_mode": "approach"})
        dock_env = ArmKinematicEnv(config=Phase1EnvConfig(mode_name="dock"))
        dock_obs, _ = dock_env.reset(seed=11, options={"policy_mode": "dock"})

        self.assertEqual(approach_obs["mode_flag"].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(dock_obs["mode_flag"].tolist(), [0.0, 1.0, 0.0, 0.0])

    def test_dock_reset_starts_near_goal(self) -> None:
        env = ArmKinematicEnv(config=Phase1EnvConfig(mode_name="dock"))
        _, info = env.reset(seed=22, options={"policy_mode": "dock"})
        self.assertLess(float(np.linalg.norm(np.asarray(info["q"]) - np.asarray(info["goal_q"]))), 0.2)

    def test_action_delta_scale_can_make_local_dock_steps_smaller(self) -> None:
        base_env = ArmKinematicEnv(config=Phase1EnvConfig(mode_name="dock", action_delta_scale=1.0))
        small_env = ArmKinematicEnv(config=Phase1EnvConfig(mode_name="dock", action_delta_scale=0.25))
        _, base_info = base_env.reset(seed=33, options={"policy_mode": "dock"})
        _, small_info = small_env.reset(seed=33, options={"policy_mode": "dock"})
        action = np.ones(7, dtype=np.float32)

        _, _, _, _, base_step_info = base_env.step(action)
        _, _, _, _, small_step_info = small_env.step(action)

        base_delta = float(np.linalg.norm(np.asarray(base_step_info["q"]) - np.asarray(base_info["q"])))
        small_delta = float(np.linalg.norm(np.asarray(small_step_info["q"]) - np.asarray(small_info["q"])))
        self.assertLess(small_delta, base_delta * 0.35)

    def test_dock_residual_action_limit_bounds_local_corrections(self) -> None:
        uncapped_env = ArmKinematicEnv(config=Phase1EnvConfig(mode_name="dock", action_delta_scale=1.0, dock_residual_action_limit=1.0))
        capped_env = ArmKinematicEnv(config=Phase1EnvConfig(mode_name="dock", action_delta_scale=1.0, dock_residual_action_limit=0.10))
        _, uncapped_info = uncapped_env.reset(seed=55, options={"policy_mode": "dock"})
        _, capped_info = capped_env.reset(seed=55, options={"policy_mode": "dock"})
        action = np.ones(7, dtype=np.float32)

        _, _, _, _, uncapped_step_info = uncapped_env.step(action)
        _, _, _, _, capped_step_info = capped_env.step(action)

        uncapped_delta = float(np.linalg.norm(np.asarray(uncapped_step_info["q"]) - np.asarray(uncapped_info["q"])))
        capped_delta = float(np.linalg.norm(np.asarray(capped_step_info["q"]) - np.asarray(capped_info["q"])))
        self.assertLess(capped_delta, uncapped_delta * 0.2)

    def test_dock_delta_q_change_limit_reduces_step_to_step_reversal(self) -> None:
        uncapped_env = ArmKinematicEnv(
            config=Phase1EnvConfig(
                mode_name="dock",
                action_delta_scale=1.0,
                dock_residual_action_limit=1.0,
                dock_delta_q_change_limit_scale=0.0,
            )
        )
        capped_env = ArmKinematicEnv(
            config=Phase1EnvConfig(
                mode_name="dock",
                action_delta_scale=1.0,
                dock_residual_action_limit=1.0,
                dock_delta_q_change_limit_scale=0.10,
            )
        )
        uncapped_env.reset(seed=77, options={"policy_mode": "dock"})
        capped_env.reset(seed=77, options={"policy_mode": "dock"})

        uncapped_env.step(np.ones(7, dtype=np.float32))
        capped_env.step(np.ones(7, dtype=np.float32))
        _, _, _, _, uncapped_step_info = uncapped_env.step(-np.ones(7, dtype=np.float32))
        _, _, _, _, capped_step_info = capped_env.step(-np.ones(7, dtype=np.float32))

        self.assertLess(capped_step_info["executed_delta_q_l2"], uncapped_step_info["executed_delta_q_l2"])

    def test_apply_dock_training_stage_updates_reset_band_and_limits(self) -> None:
        env = ArmKinematicEnv(config=Phase1EnvConfig(mode_name="dock"))
        env.apply_dock_training_stage(
            {
                "action_delta_scale": 0.02,
                "dock_residual_action_limit": 0.15,
                "dock_delta_q_change_limit_scale": 0.08,
                "dock_reset": {
                    "close_bucket_probability": 1.0,
                    "close_bucket_min_pos_error_m": 0.002,
                    "close_bucket_max_pos_error_m": 0.006,
                },
            }
        )
        self.assertAlmostEqual(env.config.action_delta_scale, 0.02)
        self.assertAlmostEqual(env.config.dock_residual_action_limit, 0.15)
        self.assertAlmostEqual(env.config.dock_delta_q_change_limit_scale, 0.08)
        self.assertAlmostEqual(env.config.dock_reset_config.close_bucket_probability, 1.0)
        self.assertAlmostEqual(env.config.dock_reset_config.close_bucket_min_pos_error_m, 0.002)
        self.assertAlmostEqual(env.config.dock_reset_config.close_bucket_max_pos_error_m, 0.006)

    def test_dock_convergence_progress_only_activates_in_inner_zone(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        config = DockRewardConfig(
            convergence_position_radius_m=0.018,
            convergence_position_progress_weight=18.0,
            convergence_orientation_radius_rad=0.12,
            convergence_orientation_progress_weight=4.0,
        )

        _, inner_components = compute_dock_reward(
            prev_pose6=np.array([0.012, 0.0, 0.0, 0.08, 0.0, 0.0], dtype=float),
            curr_pose6=np.array([0.007, 0.0, 0.0, 0.04, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )
        _, outer_components = compute_dock_reward(
            prev_pose6=np.array([0.040, 0.0, 0.0, 0.20, 0.0, 0.0], dtype=float),
            curr_pose6=np.array([0.030, 0.0, 0.0, 0.16, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            prev_in_near_goal=False,
            curr_in_near_goal=False,
            dwell_count=0,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )

        self.assertGreater(inner_components["convergence_position_progress"], 0.0)
        self.assertGreater(inner_components["convergence_orientation_progress"], 0.0)
        self.assertEqual(outer_components["convergence_position_progress"], 0.0)
        self.assertEqual(outer_components["convergence_orientation_progress"], 0.0)

    def test_dock_position_first_scales_orientation_before_position_is_ready(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        config = DockRewardConfig(
            convergence_orientation_radius_rad=0.12,
            convergence_orientation_progress_weight=4.0,
            position_first_orientation_pos_threshold_m=0.010,
            position_first_orientation_pre_scale=0.20,
        )

        _, far_components = compute_dock_reward(
            prev_pose6=np.array([0.020, 0.0, 0.0, 0.08, 0.0, 0.0], dtype=float),
            curr_pose6=np.array([0.015, 0.0, 0.0, 0.04, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )
        _, close_components = compute_dock_reward(
            prev_pose6=np.array([0.009, 0.0, 0.0, 0.08, 0.0, 0.0], dtype=float),
            curr_pose6=np.array([0.007, 0.0, 0.0, 0.04, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )

        self.assertAlmostEqual(far_components["orientation_position_gate_scale"], 0.20)
        self.assertAlmostEqual(close_components["orientation_position_gate_scale"], 1.0)
        self.assertLess(far_components["convergence_orientation_progress"], close_components["convergence_orientation_progress"])

    def test_strict_pose_leave_penalty_only_triggers_when_exiting_tight_pose(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        _, components = compute_dock_reward(
            prev_pose6=np.array([0.004, 0.0, 0.0, 0.03, 0.0, 0.0], dtype=float),
            curr_pose6=np.array([0.007, 0.0, 0.0, 0.03, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=DockRewardConfig(
                tight_pose_pos_threshold_m=0.005,
                tight_pose_ori_threshold_rad=0.05,
                strict_pose_leave_penalty=1.2,
            ),
        )
        self.assertEqual(components["strict_pose_leave_penalty"], -1.2)

    def test_strict_center_penalty_prefers_exact_center_over_edge(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        center_reward, center_components = compute_dock_reward(
            prev_pose6=np.zeros(6, dtype=float),
            curr_pose6=np.array([0.001, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=DockRewardConfig(
                tight_pose_pos_threshold_m=0.005,
                tight_pose_ori_threshold_rad=0.05,
                strict_center_position_weight=0.8,
                strict_center_orientation_weight=0.2,
            ),
        )
        edge_reward, edge_components = compute_dock_reward(
            prev_pose6=np.zeros(6, dtype=float),
            curr_pose6=np.array([0.0049, 0.0, 0.0, 0.045, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.zeros(7),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=DockRewardConfig(
                tight_pose_pos_threshold_m=0.005,
                tight_pose_ori_threshold_rad=0.05,
                strict_center_position_weight=0.8,
                strict_center_orientation_weight=0.2,
            ),
        )
        self.assertGreater(center_reward, edge_reward)
        self.assertGreater(center_components["strict_center_position_penalty"], edge_components["strict_center_position_penalty"])

    def test_strict_center_small_action_bonus_grows_near_center(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        config = DockRewardConfig(
            strict_center_small_action_bonus_weight=0.45,
            strict_center_small_action_pos_radius_m=0.0065,
            strict_center_small_action_ori_radius_rad=0.05,
            strict_center_small_action_scale=0.12,
            strict_center_small_action_power=3.0,
        )

        center_reward, center_components = compute_dock_reward(
            prev_pose6=np.zeros(6, dtype=float),
            curr_pose6=np.array([0.001, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.full(7, 0.02, dtype=float),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )
        edge_reward, edge_components = compute_dock_reward(
            prev_pose6=np.zeros(6, dtype=float),
            curr_pose6=np.array([0.0058, 0.0, 0.0, 0.04, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.full(7, 0.08, dtype=float),
            prev_action=np.zeros(7),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=1,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )

        self.assertGreater(center_reward, edge_reward)
        self.assertGreater(
            center_components["strict_center_small_action_bonus"],
            edge_components["strict_center_small_action_bonus"],
        )

    def test_strict_center_reward_and_penalties_create_regulator_behavior(self) -> None:
        goal_pose = np.zeros(6, dtype=float)
        config = DockRewardConfig(
            tight_pose_pos_threshold_m=0.005,
            tight_pose_ori_threshold_rad=0.05,
            strict_center_reward_weight=0.9,
            strict_center_position_weight=0.9,
            strict_center_orientation_weight=0.15,
            action_magnitude_weight=0.16,
            action_delta_weight=0.20,
            strict_zone_drift_penalty_multiplier=2.0,
            strict_zone_action_penalty_multiplier=1.8,
        )

        center_reward, center_components = compute_dock_reward(
            prev_pose6=np.array([0.0012, 0.0, 0.0, 0.012, 0.0, 0.0], dtype=float),
            curr_pose6=np.array([0.0010, 0.0, 0.0, 0.010, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.full(7, 0.01, dtype=float),
            prev_action=np.full(7, 0.01, dtype=float),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=3,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )
        edge_reward, edge_components = compute_dock_reward(
            prev_pose6=np.array([0.0047, 0.0, 0.0, 0.040, 0.0, 0.0], dtype=float),
            curr_pose6=np.array([0.0049, 0.0, 0.0, 0.045, 0.0, 0.0], dtype=float),
            goal_pose6=goal_pose,
            action=np.full(7, 0.04, dtype=float),
            prev_action=np.full(7, 0.02, dtype=float),
            prev_in_near_goal=True,
            curr_in_near_goal=True,
            dwell_count=3,
            joint_limit_margin_min=1.0,
            success=False,
            config=config,
        )

        self.assertGreater(center_components["strict_center_reward"], edge_components["strict_center_reward"])
        self.assertLess(center_components["smoothness_penalty"], 0.0)
        self.assertLess(edge_components["drift_penalty"], center_components["drift_penalty"])
        self.assertGreater(center_reward, edge_reward)

    def test_success_can_remain_true_without_terminating_episode(self) -> None:
        outcome = evaluate_termination(
            step_count=5,
            pos_error_norm=0.004,
            ori_error_norm=0.03,
            dwell_count=5,
            config=TerminationConfig(
                success_pos_threshold_m=0.005,
                success_ori_threshold_rad=0.05,
                success_dwell_steps=5,
                require_orientation=True,
                terminate_on_success=False,
            ),
        )
        self.assertTrue(outcome["success"])
        self.assertFalse(outcome["terminated"])
        self.assertFalse(outcome["truncated"])

    def test_success_without_termination_still_truncates_at_max_steps(self) -> None:
        outcome = evaluate_termination(
            step_count=24,
            pos_error_norm=0.004,
            ori_error_norm=0.03,
            dwell_count=5,
            config=TerminationConfig(
                max_episode_steps=24,
                success_pos_threshold_m=0.005,
                success_ori_threshold_rad=0.05,
                success_dwell_steps=5,
                require_orientation=True,
                terminate_on_success=False,
            ),
        )
        self.assertTrue(outcome["success"])
        self.assertFalse(outcome["terminated"])
        self.assertTrue(outcome["truncated"])
        self.assertEqual(outcome["reason"], "max_steps")

    def test_dock_close_bucket_reset_targets_last_centimeter_band(self) -> None:
        dock_reset_config = DockResetConfig(
            close_bucket_probability=1.0,
            close_bucket_min_pos_error_m=0.005,
            close_bucket_max_pos_error_m=0.020,
            close_bucket_max_ori_error_rad=0.12,
            close_bucket_max_attempts=256,
        )
        env = ArmKinematicEnv(
            config=Phase1EnvConfig(mode_name="dock", dock_reset_config=dock_reset_config)
        )
        _, info = env.reset(seed=44, options={"policy_mode": "dock"})
        pos_norm = float(info["position_error_norm"])
        ori_norm = float(info["orientation_error_norm"])
        self.assertGreaterEqual(pos_norm, 0.005)
        self.assertLessEqual(pos_norm, 0.020)
        self.assertLessEqual(ori_norm, 0.12)

    def test_switcher_requires_confirmed_ready_to_dock(self) -> None:
        switcher = TwoPolicySwitcher(
            SwitchingConfig(
                dock_enter_pos_threshold_m=0.08,
                dock_enter_ori_threshold_rad=0.25,
                dock_enter_dwell_steps=2,
                dock_enter_action_threshold=0.35,
                dock_enter_regression_threshold_m=0.01,
                dock_enter_confirm_steps=2,
                min_approach_steps_before_switch=2,
            )
        )
        switcher.reset()
        self.assertEqual(
            switcher.update(
                position_error_norm=0.07,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.10,
                min_position_error_so_far=0.07,
                step_index=0,
            ),
            "approach",
        )
        self.assertEqual(
            switcher.update(
                position_error_norm=0.07,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.10,
                min_position_error_so_far=0.07,
                step_index=1,
            ),
            "approach",
        )
        self.assertEqual(
            switcher.update(
                position_error_norm=0.07,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.10,
                min_position_error_so_far=0.07,
                step_index=2,
            ),
            "approach",
        )
        self.assertEqual(
            switcher.update(
                position_error_norm=0.07,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.10,
                min_position_error_so_far=0.07,
                step_index=3,
            ),
            "dock",
        )
        self.assertEqual(switcher.ready_to_dock_confirmed_count, 1)
        self.assertEqual(switcher.first_switch_step, 3)

    def test_switcher_exits_after_confirmed_timeout(self) -> None:
        switcher = TwoPolicySwitcher(
            SwitchingConfig(
                dock_enter_confirm_steps=1,
                dock_exit_confirm_steps=2,
                dock_timeout_steps=3,
                dock_timeout_min_improvement_m=0.01,
                min_approach_steps_before_switch=0,
            )
        )
        switcher.reset()
        self.assertEqual(
            switcher.update(
                position_error_norm=0.07,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.05,
                min_position_error_so_far=0.07,
                step_index=0,
            ),
            "dock",
        )
        self.assertEqual(
            switcher.update(
                position_error_norm=0.069,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.05,
                min_position_error_so_far=0.069,
                step_index=1,
            ),
            "dock",
        )
        self.assertEqual(
            switcher.update(
                position_error_norm=0.069,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.05,
                min_position_error_so_far=0.069,
                step_index=2,
            ),
            "dock",
        )
        self.assertEqual(
            switcher.update(
                position_error_norm=0.069,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.05,
                min_position_error_so_far=0.069,
                step_index=3,
            ),
            "dock",
        )
        self.assertEqual(
            switcher.update(
                position_error_norm=0.069,
                orientation_error_norm=0.20,
                dwell_count=2,
                action_magnitude=0.05,
                min_position_error_so_far=0.069,
                step_index=4,
            ),
            "approach",
        )
        self.assertEqual(switcher.dock_timeout_count, 1)
        self.assertEqual(switcher.switch_back_count, 1)
