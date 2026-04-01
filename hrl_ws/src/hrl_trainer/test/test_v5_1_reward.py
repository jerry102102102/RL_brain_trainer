import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrl_trainer.v5_1.reward import RewardComposer, RewardConfig


class RewardComposerV51Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.composer = RewardComposer(RewardConfig())
        self.zero_action = np.zeros(3, dtype=np.float32)
        self.prev_pos_err = np.array([0.03, 0.0, 0.0], dtype=np.float32)
        self.curr_dwell_pos_err = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        self.zero_ori = np.zeros(3, dtype=np.float32)

    def test_success_by_dwell_is_one_shot(self) -> None:
        awarded = self.composer.compute(
            prev_ee_pos_err=self.prev_pos_err,
            prev_ee_ori_err=self.zero_ori,
            curr_ee_pos_err=self.curr_dwell_pos_err,
            curr_ee_ori_err=self.zero_ori,
            action=self.zero_action,
            prev_action=self.zero_action,
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="",
            prev_in_near_goal=True,
            dwell_count=4,
            prev_success_latched=False,
        )

        self.assertEqual(awarded.success_bonus, self.composer.config.success_bonus)
        self.assertEqual(awarded.success_latched, 1.0)
        self.assertEqual(awarded.dwell_count, 5.0)

        suppressed = self.composer.compute(
            prev_ee_pos_err=self.curr_dwell_pos_err,
            prev_ee_ori_err=self.zero_ori,
            curr_ee_pos_err=self.curr_dwell_pos_err,
            curr_ee_ori_err=self.zero_ori,
            action=self.zero_action,
            prev_action=self.zero_action,
            intervention=False,
            clamp_or_projection=False,
            done=False,
            done_reason="",
            prev_in_near_goal=True,
            dwell_count=int(awarded.dwell_count),
            prev_success_latched=bool(awarded.success_latched),
        )

        self.assertEqual(suppressed.success_bonus, 0.0)
        self.assertEqual(suppressed.success_latched, 1.0)
        self.assertEqual(suppressed.dwell_count, 6.0)

    def test_execution_fail_clears_episode_state_and_breakdown(self) -> None:
        terms = self.composer.compute(
            prev_ee_pos_err=self.prev_pos_err,
            prev_ee_ori_err=self.zero_ori,
            curr_ee_pos_err=self.curr_dwell_pos_err,
            curr_ee_ori_err=self.zero_ori,
            action=self.zero_action,
            prev_action=self.zero_action,
            intervention=False,
            clamp_or_projection=False,
            done=True,
            done_reason="execution_fail",
            prev_in_near_goal=True,
            dwell_count=4,
            prev_success_latched=True,
        )

        self.assertEqual(terms.execution_fail_penalty, self.composer.config.execution_fail_penalty)
        self.assertEqual(terms.timeout_penalty, 0.0)
        self.assertEqual(terms.reset_fail_penalty, 0.0)
        self.assertEqual(terms.timeout_or_reset, self.composer.config.execution_fail_penalty)
        self.assertEqual(terms.in_near_goal, 0.0)
        self.assertEqual(terms.dwell_count, 0.0)
        self.assertEqual(terms.success_latched, 0.0)

    def test_terminal_breakdown_does_not_mix_success_with_penalty_bucket(self) -> None:
        terms = self.composer.compute(
            prev_ee_pos_err=self.prev_pos_err,
            prev_ee_ori_err=self.zero_ori,
            curr_ee_pos_err=self.curr_dwell_pos_err,
            curr_ee_ori_err=self.zero_ori,
            action=self.zero_action,
            prev_action=self.zero_action,
            intervention=False,
            clamp_or_projection=False,
            done=True,
            done_reason="success",
            prev_in_near_goal=True,
            dwell_count=0,
            prev_success_latched=False,
        )

        payload = terms.to_dict()
        self.assertEqual(payload["success_bonus"], self.composer.config.success_bonus)
        self.assertEqual(payload["timeout_or_reset"], 0.0)
        self.assertEqual(payload["timeout_penalty"], 0.0)
        self.assertEqual(payload["reset_fail_penalty"], 0.0)
        self.assertEqual(payload["execution_fail_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
