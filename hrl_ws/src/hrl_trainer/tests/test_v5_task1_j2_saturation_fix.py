import unittest

import numpy as np

from hrl_trainer.v5.task1_train import (
    L2Action,
    L3ExecutionResult,
    Task1Config,
    Task1State,
    compute_macro_micro_delta,
    run_task1_episode,
)


class _ConstJ2Policy:
    def decide_action(self, _obs):
        raw = np.zeros(7, dtype=float)
        raw[1] = 1.0
        return L2Action(delta_q_raw=raw)


class _ClampExecutor:
    def __init__(self, j2_limit: float = 0.02):
        self.j2_limit = float(j2_limit)

    def execute_with_safety(self, state: Task1State, delta_q_cmd: np.ndarray) -> L3ExecutionResult:
        cmd = np.asarray(delta_q_cmd, dtype=float)
        exe = cmd.copy()
        exe[1] = float(np.clip(exe[1], -self.j2_limit, self.j2_limit))
        q_next = state.q + exe
        return L3ExecutionResult(
            accepted=True,
            q_next=q_next,
            dq_next=exe,
            safety_violation=0.0,
            ee_proxy_xyz=state.ee_proxy_xyz,
            logs=("ok",),
            limited_cmd=exe.copy(),
            q_target_minus_runtime=(q_next - state.q).copy(),
            requested_delta_q=cmd.copy(),
            executed_delta_q=exe.copy(),
            feasible_ratio=1.0,
            projection_gap=0.0,
            null_effect_step=False,
            sat_ratio=0.0,
            encoder_delta=float(np.max(np.abs(exe))),
            no_motion_signal=False,
        )


class TestTask1J2SaturationFix(unittest.TestCase):
    def test_compute_macro_micro_delta_spreads_over_remaining_ttl(self):
        state_q = np.zeros(7, dtype=float)
        target_q = np.zeros(7, dtype=float)
        target_q[1] = 0.08
        dq_lim = np.array([0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=float)

        micro = compute_macro_micro_delta(
            target_q=target_q,
            state_q=state_q,
            ttl_steps_left=4,
            dq_max_per_joint=dq_lim,
        )

        self.assertAlmostEqual(float(micro[1]), 0.02, places=6)

    def test_episode_j2_saturation_ratio_drops_with_ttl_micro_step(self):
        cfg = Task1Config(
            n_joints=7,
            max_steps=4,
            safe_z_min=0.2,
            failfast_no_motion_streak=100,
        )
        dq_lim = np.array([0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=float)
        q0 = np.zeros(7, dtype=float)
        q0[2] = 0.25

        row = run_task1_episode(
            episode_index=0,
            reward_mode="task1_main",
            cfg=cfg,
            dq_max_per_joint=dq_lim,
            l1_provider=type("_L1", (), {"get_target_pose": lambda self, episode_index: np.array([0.35, 0.0, 0.35], dtype=float)})(),
            l2_policy=_ConstJ2Policy(),
            l3_executor=_ClampExecutor(j2_limit=0.02),
            initial_q=q0,
            initial_dq=np.zeros(7, dtype=float),
            initial_ee_proxy_xyz=np.array([0.0, 0.0, 0.25], dtype=float),
            episode_debug_meta={"macro_ttl_steps": 4, "backend": "gazebo", "ee_source": "test"},
        )

        # Before fix this pattern is ~0.48 (0.08/0.06/0.04/0.02 requested with 0.02 executed).
        self.assertLess(float(row["episode_summary"]["j2_saturation_ratio"]), 0.15)


if __name__ == "__main__":
    unittest.main()
