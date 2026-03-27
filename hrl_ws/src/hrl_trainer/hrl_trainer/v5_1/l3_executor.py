"""Deterministic L3 executor for V5.1.

Implements core delta_q -> q_des transform with:
- command clamp
- rate limiter
- projection to joint limits
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class L3ExecutorConfig:
    dt: float = 0.1
    joint_min: tuple[float, ...] = (-2.8, -1.6, -2.8, -3.0, -2.8, -6.0)
    joint_max: tuple[float, ...] = (2.8, 1.6, 2.8, 3.0, 2.8, 6.0)
    delta_q_limit: tuple[float, ...] = (0.05, 0.03, 0.05, 0.05, 0.05, 0.08)
    rate_limit_per_sec: tuple[float, ...] = (0.30, 0.20, 0.30, 0.30, 0.30, 0.40)


@dataclass(frozen=True)
class L3ExecutorResult:
    q_des: np.ndarray
    requested_delta_q: np.ndarray
    clamped_delta_q: np.ndarray
    limited_q_des: np.ndarray
    projection_applied: bool


class L3DeterministicExecutor:
    def __init__(self, config: L3ExecutorConfig | None = None) -> None:
        self.config = config or L3ExecutorConfig()

    def compute_q_des(
        self,
        q_current: np.ndarray,
        delta_q_cmd: np.ndarray,
        prev_q_des: np.ndarray | None = None,
    ) -> L3ExecutorResult:
        q_current = np.asarray(q_current, dtype=float)
        requested = np.asarray(delta_q_cmd, dtype=float)
        q_min = np.asarray(self.config.joint_min, dtype=float)
        q_max = np.asarray(self.config.joint_max, dtype=float)
        delta_lim = np.asarray(self.config.delta_q_limit, dtype=float)

        if not (q_current.shape == requested.shape == q_min.shape == q_max.shape == delta_lim.shape):
            raise ValueError("shape mismatch in L3 deterministic executor")

        clamped_delta = np.clip(requested, -delta_lim, delta_lim)
        pre_rate_q_des = q_current + clamped_delta

        if prev_q_des is not None:
            prev_q_des = np.asarray(prev_q_des, dtype=float)
            if prev_q_des.shape != q_current.shape:
                raise ValueError("prev_q_des shape mismatch")
            max_step = np.asarray(self.config.rate_limit_per_sec, dtype=float) * float(self.config.dt)
            limited_step = np.clip(pre_rate_q_des - prev_q_des, -max_step, max_step)
            limited_q_des = prev_q_des + limited_step
        else:
            limited_q_des = pre_rate_q_des

        projected = np.clip(limited_q_des, q_min, q_max)
        projection_applied = bool(np.any(np.abs(projected - limited_q_des) > 1e-12))

        return L3ExecutorResult(
            q_des=projected,
            requested_delta_q=requested,
            clamped_delta_q=clamped_delta,
            limited_q_des=limited_q_des,
            projection_applied=projection_applied,
        )
