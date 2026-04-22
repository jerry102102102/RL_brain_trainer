"""Pose-space utilities for the Phase 1 kinematic environment."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wrap_to_pi(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def orientation_error_rpy(curr_rpy: Sequence[float], goal_rpy: Sequence[float]) -> np.ndarray:
    curr = np.asarray(curr_rpy, dtype=float)
    goal = np.asarray(goal_rpy, dtype=float)
    return np.asarray(wrap_to_pi(goal - curr), dtype=float)


def pose_error_components(curr_pose6: Sequence[float], goal_pose6: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    curr = np.asarray(curr_pose6, dtype=float)
    goal = np.asarray(goal_pose6, dtype=float)
    pos_err = goal[:3] - curr[:3]
    ori_err = orientation_error_rpy(curr[3:], goal[3:])
    return pos_err, ori_err


def l2_norm(vec: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=float)))


def normalize_vector(vec: Sequence[float], scale: float) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return np.clip(np.asarray(vec, dtype=float) / scale, -1.0, 1.0)
