from __future__ import annotations

import numpy as np

from hrl_trainer.v5_1.l3_executor import L3DeterministicExecutor, L3ExecutorConfig


def test_l3_executor_clamp_and_projection() -> None:
    cfg = L3ExecutorConfig(
        dt=0.1,
        joint_min=(-0.1,) * 6,
        joint_max=(0.1,) * 6,
        delta_q_limit=(0.05,) * 6,
        rate_limit_per_sec=(10.0,) * 6,
    )
    exe = L3DeterministicExecutor(cfg)
    q = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    result = exe.compute_q_des(q_current=q, delta_q_cmd=np.array([0.5, 0, 0, 0, 0, 0], dtype=float))

    assert np.isclose(result.clamped_delta_q[0], 0.05)
    assert np.isclose(result.q_des[0], 0.1)
    assert result.projection_applied is True


def test_l3_executor_rate_limit() -> None:
    cfg = L3ExecutorConfig(
        dt=0.1,
        joint_min=(-2.0,) * 6,
        joint_max=(2.0,) * 6,
        delta_q_limit=(1.0,) * 6,
        rate_limit_per_sec=(0.2,) * 6,
    )
    exe = L3DeterministicExecutor(cfg)
    q = np.zeros(6, dtype=float)
    prev_q_des = np.zeros(6, dtype=float)

    result = exe.compute_q_des(q_current=q, delta_q_cmd=np.array([1.0, 0, 0, 0, 0, 0], dtype=float), prev_q_des=prev_q_des)

    assert np.isclose(result.q_des[0], 0.02)
    assert np.isclose(result.limited_q_des[0], 0.02)
