from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.v5_1.reward import RewardComposer


class TestV51Reward(unittest.TestCase):
    def test_reward_components_have_expected_signs(self) -> None:
        composer = RewardComposer()
        terms = composer.compute(
            prev_ee_pos_err=np.array([0.5,0,0]),
            prev_ee_ori_err=np.array([0.0,0,0]),
            curr_ee_pos_err=np.array([0.4,0,0]),
            curr_ee_ori_err=np.array([0.0,0,0]),
            action=np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
            prev_action=np.zeros(6),
            intervention=True,
            clamp_or_projection=True,
            done=True,
            done_reason="timeout",
        )

        self.assertGreater(terms.progress, 0.0)
        self.assertLess(terms.action, 0.0)
        self.assertLess(terms.jerk, 0.0)
        self.assertLess(terms.intervention, 0.0)
        self.assertLess(terms.clamp_or_projection, 0.0)
        self.assertEqual(terms.timeout_or_reset, 0.0)
        self.assertEqual(terms.success_bonus, 0.0)

    def test_reward_success_bonus_applied(self) -> None:
        composer = RewardComposer()
        terms = composer.compute(
            prev_ee_pos_err=np.array([0.2,0,0]),
            prev_ee_ori_err=np.array([0.0,0,0]),
            curr_ee_pos_err=np.array([0.01,0,0]),
            curr_ee_ori_err=np.array([0.0,0,0]),
            action=np.zeros(6),
            prev_action=np.zeros(6),
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
            action=np.ones(6),
            prev_action=np.zeros(6),
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
        self.assertEqual(terms.timeout_or_reset, -2.0)

    def test_reward_trace_component_schema(self) -> None:
        terms = RewardComposer().compute(
            prev_ee_pos_err=np.array([0.2,0,0]),
            prev_ee_ori_err=np.array([0.0,0,0]),
            curr_ee_pos_err=np.array([0.18,0,0]),
            curr_ee_ori_err=np.array([0.0,0,0]),
            action=np.zeros(6),
            prev_action=np.zeros(6),
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
                "timeout_or_reset",
                "success_bonus",
                "reward_total",
            },
        )


if __name__ == "__main__":
    unittest.main()
