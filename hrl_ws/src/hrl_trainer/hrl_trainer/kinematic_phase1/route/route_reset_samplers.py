"""Reset samplers for route curriculum training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kinematics.joint_limits import JointSpec, clip_joint_configuration
from .route_dataset import RouteDataset


@dataclass(frozen=True)
class RouteResetSamplerConfig:
    mode: str = "mixed_prefix_segment"
    min_route_index: int = 1
    max_route_index: int = 20
    segment_start_index: int = 1
    segment_end_index: int = 40
    replay_start_index: int = 1
    replay_end_index: int = 120
    prefix_start_reset_ratio: float = 0.10
    random_prefix_reset_ratio: float = 0.55
    segment_reset_ratio: float = 0.20
    replay_reset_ratio: float = 0.0
    recovery_reset_ratio: float = 0.15
    q_noise_std: float = 0.002
    dq_noise_std: float = 0.0005
    prev_action_noise_std: float = 0.02


@dataclass(frozen=True)
class RouteResetSample:
    initial_q: np.ndarray
    initial_dq: np.ndarray
    initial_prev_action: np.ndarray
    goal_q: np.ndarray
    route_index: int
    start_route_index: int
    reset_mode: str


def _normal_noise(rng: np.random.Generator, shape: tuple[int, ...], std: float) -> np.ndarray:
    return rng.normal(0.0, float(std), size=shape) if std > 0.0 else np.zeros(shape, dtype=float)


def sample_route_reset(
    *,
    rng: np.random.Generator,
    route: RouteDataset,
    joint_specs: list[JointSpec] | tuple[JointSpec, ...],
    config: RouteResetSamplerConfig,
) -> RouteResetSample:
    max_index = len(route) - 1
    lo = int(np.clip(config.min_route_index, 1, max_index))
    hi = int(np.clip(config.max_route_index, lo, max_index))

    ratios = np.asarray(
        [
            max(config.prefix_start_reset_ratio, 0.0),
            max(config.random_prefix_reset_ratio, 0.0),
            max(config.segment_reset_ratio, 0.0),
            max(config.replay_reset_ratio, 0.0),
            max(config.recovery_reset_ratio, 0.0),
        ],
        dtype=float,
    )
    ratios = ratios / ratios.sum() if ratios.sum() > 0.0 else np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    mode = str(rng.choice(["prefix_start", "random_prefix", "segment", "replay", "recovery"], p=ratios))

    if config.mode == "prefix_start_reset":
        mode = "prefix_start"
    elif config.mode == "random_prefix_reset":
        mode = "random_prefix"
    elif config.mode == "segment_reset":
        mode = "segment"
    elif config.mode == "replay_reset":
        mode = "replay"
    elif config.mode == "recovery_reset":
        mode = "recovery"

    if mode == "prefix_start":
        route_index = int(rng.integers(lo, hi + 1))
        start_index = 0
    elif mode == "segment":
        seg_lo = int(np.clip(config.segment_start_index, 1, max_index))
        seg_hi = int(np.clip(config.segment_end_index, seg_lo, max_index))
        route_index = int(rng.integers(seg_lo, min(seg_hi, hi) + 1))
        start_index = max(route_index - 1, 0)
    elif mode == "replay":
        replay_lo = int(np.clip(config.replay_start_index, 1, max_index))
        replay_hi = int(np.clip(config.replay_end_index, replay_lo, max_index))
        route_index = int(rng.integers(replay_lo, min(replay_hi, hi) + 1))
        start_index = max(route_index - 1, 0)
    else:
        route_index = int(rng.integers(lo, hi + 1))
        start_index = max(route_index - 1, 0)

    goal_q = route.waypoint(route_index).q_goal.copy()
    initial_q = route.waypoint(start_index).q_goal.copy()
    if mode == "recovery":
        initial_q = route.waypoint(route_index).q_goal.copy()

    initial_q = initial_q + _normal_noise(rng, initial_q.shape, config.q_noise_std)
    initial_q = clip_joint_configuration(initial_q, joint_specs)
    initial_dq = _normal_noise(rng, initial_q.shape, config.dq_noise_std)
    initial_prev_action = np.clip(_normal_noise(rng, initial_q.shape, config.prev_action_noise_std), -1.0, 1.0)

    return RouteResetSample(
        initial_q=initial_q,
        initial_dq=initial_dq,
        initial_prev_action=initial_prev_action,
        goal_q=goal_q,
        route_index=route_index,
        start_route_index=start_index,
        reset_mode=mode,
    )
