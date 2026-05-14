"""Route-specific observation augmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.spaces import Box, Dict
from ..kinematics.joint_limits import JointSpec, normalize_joint_deltas, normalize_joint_positions


@dataclass(frozen=True)
class RouteObservationConfig:
    include_route_keys: bool = False


def build_route_observation_space(base_space: Dict, n_joints: int) -> Dict:
    spaces = dict(base_space.spaces)
    spaces.update(
        {
            "route_q_goal": Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32),
            "route_q_error": Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32),
            "route_tangent": Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32),
            "route_scalar": Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
        }
    )
    return Dict(spaces)


def augment_route_observation(
    obs: dict[str, np.ndarray],
    *,
    q: np.ndarray,
    q_goal: np.ndarray,
    route_tangent_q: np.ndarray,
    route_index: int,
    max_route_index: int,
    route_progress_m: float,
    total_route_progress_m: float,
    joint_specs: tuple[JointSpec, ...] | list[JointSpec],
    enabled: bool,
) -> dict[str, np.ndarray]:
    if not enabled:
        return obs
    out = dict(obs)
    q_arr = np.asarray(q, dtype=float)
    goal_arr = np.asarray(q_goal, dtype=float)
    tangent = np.asarray(route_tangent_q, dtype=float)
    out["route_q_goal"] = normalize_joint_positions(goal_arr, joint_specs).astype(np.float32)
    out["route_q_error"] = normalize_joint_deltas(goal_arr - q_arr, joint_specs).astype(np.float32)
    out["route_tangent"] = normalize_joint_deltas(tangent, joint_specs).astype(np.float32)
    out["route_scalar"] = np.array(
        [
            float(np.clip(route_index / max(max_route_index, 1), 0.0, 1.0)),
            float(np.clip(route_progress_m / max(total_route_progress_m, 1e-9), 0.0, 1.0)),
            0.0,
        ],
        dtype=np.float32,
    )
    return out
