from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.v5_1.reward import RewardComposer, RewardConfig


class TestV51Reward(unittest.TestCase):
    def test_default_progress_weight_is_boosted(self) -> None:
        self.assertEqual(RewardConfig.w_progress, 3.0)

    def test_reward_components_have_expected_signs(self) -> None:
        composer = RewardComposer()
        terms = composer.compute(
            prev_ee_pos_err=np.array([0.5,0,0]),
            prev_ee_ori_err=np.array([0.2,0,0]),
            curr_ee_pos_err=np.array([0.4,0,0]),
            curr_ee_ori_err=np.array([0.18,0,0]),
            action=np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
            prev_action=np.zeros(7),
            intervention=True,
            clamp_or_projection=True,
            done=True,
            done_reason="timeout",
        )

        self.assertGreater(terms.progress, 0.0)
        self.assertEqual(terms.action, 0.0)
        self.assertEqual(terms.jerk, 0.0)
        self.assertLess(terms.intervention, 0.0)
        self.assertLess(terms.clamp_or_projection, 0.0)
        self.assertEqual(terms.stall, 0.0)
        self.assertEqual(terms.ee_small_motion_penalty, 0.0)
        self.assertEqual(terms.timeout_or_reset, 0.0)
        self.assertEqual(terms.success_bonus, 0.0)

    def test_reward_success_bonus_applied(self) -> None:
        composer = RewardComposer()
        terms = composer.compute(
            prev_ee_pos_err=np.array([0.2,0,0]),
            prev_ee_ori_err=np.array([0.0,0,0]),
            curr_ee_pos_err=np.array([0.01,0,0]),
            curr_ee_ori_err=np.array([0.0,0,0]),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=True,
            done_reason="success",
        )
        self.assertEqual(terms.success_bonus, 1.5)

    def test_execution_fail_applies_only_fail_penalty(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2,0,0]),
            prev_ee_ori_err=np.array([0.0,0,0]),
            curr_ee_pos_err=np.array([0.1,0,0]),
            curr_ee_ori_err=np.array([0.0,0,0]),
            action=np.ones(7),
            prev_action=np.zeros(7),
            intervention=True,
            clamp_or_projection=True,
            done=True,
            done_reason="execution_fail",
        )
        self.assertEqual(terms.progress, 0.0)
        self.assertEqual(terms.action, 0.0)
        self.assertEqual(terms.jerk, 0.0)
        self.assertEqual(terms.intervention, 0.0)
        self.assertEqual(terms.clamp_or_projection, 0.0)
        self.assertEqual(terms.stall, 0.0)
        self.assertEqual(terms.ee_small_motion_penalty, 0.0)
        self.assertEqual(terms.timeout_or_reset, -2.0)

    def test_reward_trace_component_schema(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2,0,0]),
            prev_ee_ori_err=np.array([0.0,0,0]),
            curr_ee_pos_err=np.array([0.18,0,0]),
            curr_ee_ori_err=np.array([0.0,0,0]),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
        )
        payload = terms.to_dict()
        self.assertEqual(
            set(payload.keys()),
            {
                "progress",
                "action",
                "jerk",
                "intervention",
                "clamp_or_projection",
                "stall",
                "ee_small_motion_penalty",
                "timeout_or_reset",
                "success_bonus",
                "reward_total",
            },
        )

    def test_stall_penalty_triggers_on_small_joint_delta(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2, 0.0, 0.0]),
            prev_ee_ori_err=np.array([0.0, 0.0, 0.0]),
            curr_ee_pos_err=np.array([0.19, 0.0, 0.0]),
            curr_ee_ori_err=np.array([0.0, 0.0, 0.0]),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
            q_before=np.zeros(7),
            q_after=np.full(7, 1e-6),
        )
        self.assertLess(terms.stall, 0.0)

    def test_stall_penalty_not_triggered_when_joint_delta_large_enough(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2, 0.0, 0.0]),
            prev_ee_ori_err=np.array([0.0, 0.0, 0.0]),
            curr_ee_pos_err=np.array([0.19, 0.0, 0.0]),
            curr_ee_ori_err=np.array([0.0, 0.0, 0.0]),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
            q_before=np.zeros(7),
            q_after=np.full(7, 1e-2),
            effect_ratio=1.0,
        )
        self.assertEqual(terms.stall, 0.0)

    def test_stall_penalty_triggers_on_low_effect_ratio_threshold(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2, 0.0, 0.0]),
            prev_ee_ori_err=np.array([0.0, 0.0, 0.0]),
            curr_ee_pos_err=np.array([0.19, 0.0, 0.0]),
            curr_ee_ori_err=np.array([0.0, 0.0, 0.0]),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
            q_before=np.zeros(7),
            q_after=np.full(7, 1e-2),
            effect_ratio=0.2,
        )
        self.assertLess(terms.stall, 0.0)

    def test_ee_small_motion_penalty_triggers_when_step_motion_too_small(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2, 0.0, 0.0]),
            prev_ee_ori_err=np.array([0.1, 0.0, 0.0]),
            curr_ee_pos_err=np.array([0.1995, 0.0, 0.0]),
            curr_ee_ori_err=np.array([0.0998, 0.0, 0.0]),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
            effect_ratio=1.0,
        )
        self.assertLess(terms.ee_small_motion_penalty, 0.0)

    def test_ee_small_motion_penalty_not_triggered_when_motion_sufficient(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2, 0.0, 0.0]),
            prev_ee_ori_err=np.array([0.1, 0.0, 0.0]),
            curr_ee_pos_err=np.array([0.195, 0.0, 0.0]),
            curr_ee_ori_err=np.array([0.09, 0.0, 0.0]),
            action=np.zeros(7),
            prev_action=np.zeros(7),
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="running",
            effect_ratio=1.0,
        )
        self.assertEqual(terms.ee_small_motion_penalty, 0.0)


if __name__ == "__main__":
    unittest.main()
