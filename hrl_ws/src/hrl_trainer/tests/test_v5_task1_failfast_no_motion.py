import unittest

import numpy as np

from hrl_trainer.v5.task1_train import (
    HighPoseTargetProvider,
    L2Action,
    L3ExecutionResult,
    Task1Config,
    Task1Observation,
    Task1State,
    run_task1_episode,
)


class _StaticL2Policy:
    def __init__(self, n_joints: int):
        self._n_joints = int(n_joints)

    def decide_action(self, obs: Task1Observation) -> L2Action:
        _ = obs
        return L2Action(delta_q_raw=np.zeros(self._n_joints, dtype=float))


class _NoMotionAcceptedL3:
    def execute_with_safety(self, state: Task1State, delta_q_cmd: np.ndarray) -> L3ExecutionResult:
        _ = delta_q_cmd
        return L3ExecutionResult(
            accepted=True,
            q_next=state.q.copy(),
            dq_next=np.zeros_like(state.q),
            safety_violation=0.0,
            logs=("L3_EXEC:accepted", "NO_MOTION_STREAK"),
            limited_cmd=np.zeros_like(state.q),
            q_target_minus_runtime=np.zeros_like(state.q),
            requested_delta_q=np.zeros_like(state.q),
            executed_delta_q=np.zeros_like(state.q),
            feasible_ratio=1.0,
            projection_gap=0.0,
            null_effect_step=False,
            sat_ratio=0.0,
            encoder_delta=0.0,
            no_motion_signal=True,
        )


class TestTask1FailfastNoMotion(unittest.TestCase):
    def test_failfast_no_motion_streak_terminates_episode(self):
        cfg = Task1Config(
            n_joints=6,
            max_steps=20,
            failfast_no_motion_streak=3,
            epsilon_motion=1e-3,
        )

        row = run_task1_episode(
            episode_index=0,
            reward_mode="task1_main",
            cfg=cfg,
            dq_max_per_joint=np.full(cfg.n_joints, 0.03, dtype=float),
            l1_provider=HighPoseTargetProvider(target_xyz=np.array([0.35, 0.0, 0.35], dtype=float)),
            l2_policy=_StaticL2Policy(cfg.n_joints),
            l3_executor=_NoMotionAcceptedL3(),
            initial_q=np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=float),
            initial_dq=np.zeros(cfg.n_joints, dtype=float),
            initial_ee_proxy_xyz=np.array([0.0, 0.0, 0.25], dtype=float),
        )

        self.assertFalse(row["success"])
        self.assertEqual(row["term_reason"], "failfast_no_motion_streak")
        self.assertEqual(row["steps"], 3)

        summary = row["episode_summary"]
        self.assertTrue(summary["failfast_no_motion_triggered"])
        self.assertEqual(summary["failfast_no_motion_streak_threshold"], 3)
        self.assertEqual(summary["accepted_no_motion_streak_max"], 3)
        self.assertEqual(summary["term_reason_counts"].get("failfast_no_motion_streak"), 1)


if __name__ == "__main__":
    unittest.main()
