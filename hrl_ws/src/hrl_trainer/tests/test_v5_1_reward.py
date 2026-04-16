from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.v5_1.reward import RewardComposer, RewardConfig


def _vec(x: float) -> np.ndarray:
    return np.array([x, 0.0, 0.0], dtype=float)


class TestV51Reward(unittest.TestCase):
    def test_reward_config_defaults_match_phase1_ablation(self) -> None:
        cfg = RewardConfig()
        self.assertEqual(cfg.w_pos_progress_lin_toward, 6.0)
        self.assertEqual(cfg.w_pos_progress_lin_away, 9.0)
        self.assertEqual(cfg.w_pos_progress_away_near_scale, 3.0)
        self.assertEqual(cfg.w_clamp_projection, -0.12)
        self.assertEqual(cfg.near_goal_bonus, 0.03)
        self.assertEqual(cfg.outer_shell_pos_m, 0.08)
        self.assertEqual(cfg.inner_shell_pos_m, 0.04)
        self.assertEqual(cfg.near_goal_shell_pos_m, 0.08)
        self.assertEqual(cfg.shell_bonus, 0.05)
        self.assertEqual(cfg.inner_shell_bonus, 0.10)
        self.assertEqual(cfg.drift_lambda, 8.0)
        self.assertEqual(cfg.outer_exit_penalty, -0.10)
        self.assertEqual(cfg.inner_exit_penalty, -0.20)
        self.assertEqual(cfg.near_goal_exit_penalty, -0.20)
        self.assertEqual(cfg.dwell_pos_m, 0.025)
        self.assertEqual(cfg.dwell_bonus, 0.30)
        self.assertEqual(cfg.success_dwell_steps, 3)
        self.assertEqual(cfg.dwell_steps_required, 3)
        self.assertEqual(cfg.dwell_break_penalty, -0.30)
        self.assertEqual(cfg.w_adjust, 0.05)
        self.assertEqual(cfg.w_raw_action, 0.01)
        self.assertEqual(cfg.reject_penalty, -0.5)
        self.assertEqual(cfg.reject_delta_threshold, 0.8)

    def test_dwell_is_position_only_even_with_large_orientation_error(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()

        terms = composer.compute(
            prev_ee_pos_err=_vec(0.030),
            prev_ee_ori_err=_vec(0.50),
            curr_ee_pos_err=_vec(0.020),
            curr_ee_ori_err=_vec(0.50),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        self.assertEqual(terms.in_dwell, 1.0)
        self.assertEqual(terms.dwell, 0.30)
        self.assertEqual(terms.dwell_count, 1.0)

    def test_near_goal_bonus_is_entry_only(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()

        first = composer.compute(
            prev_ee_pos_err=_vec(0.050),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.035),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        second = composer.compute(
            prev_ee_pos_err=_vec(0.035),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.030),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        self.assertEqual(first.near_goal, 0.03)
        self.assertEqual(second.near_goal, 0.0)
        self.assertGreater(first.inner_shell, 0.0)
        self.assertGreater(second.inner_shell, first.inner_shell)

    def test_near_goal_shell_bonus_gets_stronger_when_closer_in_outer_shell(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()

        first = composer.compute(
            prev_ee_pos_err=_vec(0.090),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.070),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        second = composer.compute(
            prev_ee_pos_err=_vec(0.070),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.050),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        self.assertGreater(first.near_goal_shell, 0.0)
        self.assertGreater(second.near_goal_shell, first.near_goal_shell)
        self.assertEqual(first.in_near_goal_shell, 1.0)
        self.assertEqual(second.in_near_goal_shell, 1.0)
        self.assertEqual(first.in_inner_shell, 0.0)
        self.assertEqual(first.dwell, 0.0)
        self.assertEqual(second.dwell, 0.0)
        self.assertEqual(first.in_near_goal, 0.0)
        self.assertEqual(second.in_near_goal, 0.0)

    def test_local_drift_penalty_fires_when_drifting_within_shell(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()

        composer.compute(
            prev_ee_pos_err=_vec(0.090),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.060),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        drift = composer.compute(
            prev_ee_pos_err=_vec(0.060),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.070),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        self.assertLess(drift.local_drift_penalty, 0.0)
        self.assertEqual(drift.near_goal_exit, 0.0)
        self.assertLessEqual(drift.local_drift_penalty, -0.08)

    def test_away_progress_is_penalized_more_than_toward_progress(self) -> None:
        composer = RewardComposer()
        toward = composer.compute(
            prev_ee_pos_err=_vec(0.060),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.050),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        composer.reset_episode_state()
        away = composer.compute(
            prev_ee_pos_err=_vec(0.050),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.060),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        self.assertGreater(toward.progress, 0.0)
        self.assertLess(away.progress, 0.0)
        self.assertGreater(abs(away.progress), toward.progress)

    def test_away_progress_is_penalized_more_when_closer_to_goal(self) -> None:
        composer = RewardComposer()
        far_away = composer.compute(
            prev_ee_pos_err=_vec(0.080),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.090),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        composer.reset_episode_state()
        near_away = composer.compute(
            prev_ee_pos_err=_vec(0.015),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.025),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        self.assertLess(far_away.progress, 0.0)
        self.assertLess(near_away.progress, 0.0)
        self.assertGreater(abs(near_away.progress), abs(far_away.progress))

    def test_dwell_break_penalty_and_near_goal_exit_penalty_fire_on_exit(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()

        composer.compute(
            prev_ee_pos_err=_vec(0.030),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.020),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        exit_terms = composer.compute(
            prev_ee_pos_err=_vec(0.020),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.050),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        self.assertEqual(exit_terms.near_goal_exit, 0.0)
        self.assertEqual(exit_terms.inner_exit, 0.0)
        self.assertEqual(exit_terms.zone_exit, -0.30)
        self.assertEqual(exit_terms.dwell_break, -0.30)
        self.assertEqual(exit_terms.dwell_count, 0.0)

    def test_outer_shell_exit_penalty_fires_when_leaving_outer_shell(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()
        composer.compute(
            prev_ee_pos_err=_vec(0.090),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.060),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        exit_terms = composer.compute(
            prev_ee_pos_err=_vec(0.060),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.090),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        self.assertEqual(exit_terms.outer_exit, -0.10)
        self.assertEqual(exit_terms.zone_exit, -0.10)

    def test_success_bonus_is_latched_after_dwell_trigger(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()

        for prev_dpos, curr_dpos in ((0.030, 0.020), (0.020, 0.018)):
            terms = composer.compute(
                prev_ee_pos_err=_vec(prev_dpos),
                prev_ee_ori_err=_vec(0.0),
                curr_ee_pos_err=_vec(curr_dpos),
                curr_ee_ori_err=_vec(0.0),
                action=np.zeros(7),
                prev_action=np.zeros(7),
                intervention=False,
                clamp_or_projection=False,
                done=False,
                done_reason="running",
            )
            self.assertEqual(terms.success_bonus, 0.0)

        triggered = composer.compute(
            prev_ee_pos_err=_vec(0.018),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.015),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        latched_terminal = composer.compute(
            prev_ee_pos_err=_vec(0.015),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.010),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=True,
            done_reason="success",
        )

        self.assertEqual(triggered.success_bonus, 3.0)
        self.assertEqual(triggered.success_triggered_by_dwell, 1.0)
        self.assertEqual(triggered.success_latched, 1.0)
        self.assertEqual(latched_terminal.success_bonus, 0.0)
        self.assertEqual(latched_terminal.success_latched, 1.0)

    def test_execution_fail_clears_episode_state_and_returns_only_fail_penalty(self) -> None:
        composer = RewardComposer()
        composer.reset_episode_state()
        composer.compute(
            prev_ee_pos_err=_vec(0.030),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.020),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )

        terms = composer.compute(
            prev_ee_pos_err=_vec(0.020),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.040),
            curr_ee_ori_err=_vec(0.0),
            action=np.ones(7),
            prev_action=np.zeros(7),
            intervention=True,
            clamp_or_projection=True,
            done=True,
            done_reason="execution_fail",
        )

        self.assertEqual(terms.reward_total, -2.0)
        self.assertEqual(terms.timeout_or_reset, -2.0)

    def test_adjust_and_raw_action_penalties_follow_raw_vs_exec_gap(self) -> None:
        composer = RewardComposer()
        terms = composer.compute(
            prev_ee_pos_err=_vec(0.050),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.045),
            curr_ee_ori_err=_vec(0.0),
            action=np.array([0.01] * 7, dtype=float),
            action_raw=np.array([0.05] * 7, dtype=float),
            action_exec=np.array([0.01] * 7, dtype=float),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=True,
            rejected=False,
            done=False,
            done_reason="running",
        )

        self.assertLess(terms.adjust_penalty, 0.0)
        self.assertLess(terms.raw_action_penalty, 0.0)
        self.assertEqual(terms.reject_penalty, 0.0)

    def test_reject_penalty_fires_when_rejected(self) -> None:
        composer = RewardComposer()
        terms = composer.compute(
            prev_ee_pos_err=_vec(0.050),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.060),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            action_raw=np.array([0.05] * 7, dtype=float),
            action_exec=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            rejected=True,
            done=False,
            done_reason="running",
        )

        self.assertEqual(terms.reject_penalty, -0.5)
        self.assertEqual(terms.dwell_break, 0.0)
        self.assertEqual(composer.state_dict()["dwell_count"], 0)
        self.assertFalse(bool(composer.state_dict()["success_awarded"]))

    def test_reward_trace_schema_includes_new_reward_fields(self) -> None:
        composer = RewardComposer()
        terms = composer.compute(
            prev_ee_pos_err=_vec(0.050),
            prev_ee_ori_err=_vec(0.0),
            curr_ee_pos_err=_vec(0.030),
            curr_ee_ori_err=_vec(0.0),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=True,
            done=False,
            done_reason="running",
        )
        payload = terms.to_dict()

        for key in (
            "progress",
            "clamp_or_projection",
            "near_goal",
            "inner_shell",
            "outer_exit",
            "inner_exit",
            "zone_exit",
            "dwell",
            "near_goal_exit",
            "dwell_break",
            "in_near_goal",
            "in_near_goal_shell",
            "in_inner_shell",
            "in_dwell",
            "zone_index",
            "dwell_count",
            "success_triggered_by_dwell",
            "success_latched",
            "reward_total",
        ):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
