"""Gymnasium space helpers with lightweight fallbacks.

The fallbacks keep this package importable even before Gymnasium is
installed in the active environment. SB3 training still requires the real
Gymnasium package at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
    EnvBase = gym.Env
    Box = spaces.Box
    Dict = spaces.Dict
except Exception:  # pragma: no cover - fallback path used in local tests
    gym = None
    GYMNASIUM_AVAILABLE = False

    class EnvBase:
        """Tiny fallback base class mirroring the Gymnasium API surface."""

        metadata: dict[str, Any] = {}

    @dataclass
    class Box:
        low: np.ndarray
        high: np.ndarray
        shape: tuple[int, ...]
        dtype: Any = np.float32

        def __init__(
            self,
            low: float | np.ndarray,
            high: float | np.ndarray,
            shape: tuple[int, ...] | None = None,
            dtype: Any = np.float32,
        ) -> None:
            self.dtype = dtype
            if shape is None:
                low_arr = np.asarray(low, dtype=dtype)
                high_arr = np.asarray(high, dtype=dtype)
                self.shape = tuple(low_arr.shape)
            else:
                low_arr = np.full(shape, low, dtype=dtype)
                high_arr = np.full(shape, high, dtype=dtype)
                self.shape = shape
            self.low = low_arr
            self.high = high_arr

    @dataclass
    class Dict:
        spaces: dict[str, Box]

    class _SpacesModule:
        Box = Box
        Dict = Dict

    spaces = _SpacesModule()


def build_action_space(n_joints: int) -> Box:
    return Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)


def build_observation_space(n_joints: int) -> Dict:
    return Dict(
        {
            "q": Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32),
            "dq": Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32),
            "prev_action": Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32),
            "goal_pos_err": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "goal_ori_err": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "wp_pos_err": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "wp_ori_err": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "next_wp_pos_err": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "next_wp_ori_err": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "task_type": Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "mode_flag": Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "progress": Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "joint_limit_margin": Box(low=0.0, high=1.0, shape=(n_joints,), dtype=np.float32),
        }
    )
