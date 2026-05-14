"""Reset samplers for approach and dock Phase 1B modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .curriculum import PointCurriculumConfig, sample_stage_joint_target
from ..kinematics.fk_interface import compute_ee_pose6
from ..kinematics.joint_limits import JointSpec, clip_joint_configuration, sample_joint_configuration
from ..kinematics.pose_utils import l2_norm, pose_error_components


def _tuple7(values: Sequence[float]) -> tuple[float, ...]:
    data = tuple(float(v) for v in values)
    if len(data) != 7:
        raise ValueError("Phase 1B reset samplers require 7-joint vectors")
    return data


def _tuple6(values: Sequence[float]) -> tuple[float, ...]:
    data = tuple(float(v) for v in values)
    if len(data) != 6:
        raise ValueError("Phase 1B reset samplers require 6D pose vectors")
    return data


@dataclass(frozen=True)
class HandoffResetState:
    initial_q: tuple[float, ...]
    goal_q: tuple[float, ...]
    goal_pose6: tuple[float, ...]
    initial_dq: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    initial_prev_action: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_q", _tuple7(self.initial_q))
        object.__setattr__(self, "goal_q", _tuple7(self.goal_q))
        object.__setattr__(self, "goal_pose6", _tuple6(self.goal_pose6))
        object.__setattr__(self, "initial_dq", _tuple7(self.initial_dq))
        object.__setattr__(self, "initial_prev_action", _tuple7(self.initial_prev_action))


@dataclass(frozen=True)
class DockResetConfig:
    goal_q: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    goal_noise: tuple[float, ...] = (0.01, 0.03, 0.04, 0.03, 0.02, 0.02, 0.01)
    init_q_noise: tuple[float, ...] = (0.01, 0.02, 0.03, 0.02, 0.015, 0.015, 0.01)
    close_bucket_probability: float = 0.0
    close_init_q_noise: tuple[float, ...] = (0.006, 0.012, 0.018, 0.012, 0.009, 0.009, 0.006)
    close_bucket_min_pos_error_m: float = 0.005
    close_bucket_max_pos_error_m: float = 0.020
    close_bucket_min_ori_error_rad: float = 0.0
    close_bucket_max_ori_error_rad: float = 0.12
    close_bucket_max_attempts: int = 128
    handoff_state_probability: float = 0.0
    handoff_state_buffer_path: str = ""
    handoff_state_max_position_error_m: float = 1.0
    handoff_state_max_orientation_error_rad: float = 10.0
    handoff_state_max_action_l2: float = 10.0
    _handoff_states: tuple[HandoffResetState, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_q", _tuple7(self.goal_q))
        object.__setattr__(self, "goal_noise", _tuple7(self.goal_noise))
        object.__setattr__(self, "init_q_noise", _tuple7(self.init_q_noise))
        object.__setattr__(self, "close_init_q_noise", _tuple7(self.close_init_q_noise))
        object.__setattr__(self, "handoff_state_buffer_path", str(self.handoff_state_buffer_path or ""))
        object.__setattr__(
            self,
            "_handoff_states",
            _load_handoff_states(
                self.handoff_state_buffer_path,
                max_position_error_m=float(self.handoff_state_max_position_error_m),
                max_orientation_error_rad=float(self.handoff_state_max_orientation_error_rad),
                max_action_l2=float(self.handoff_state_max_action_l2),
            ),
        )


@dataclass(frozen=True)
class RouteResetConfig:
    enabled: bool = False
    route_path: str = ""
    min_stride_by_stage: tuple[int, ...] = (1,)
    max_stride_by_stage: tuple[int, ...] = (1,)
    start_q_noise: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    goal_q_noise: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reverse_probability: float = 0.0
    _route_q: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_path", str(self.route_path or ""))
        object.__setattr__(self, "min_stride_by_stage", tuple(int(v) for v in self.min_stride_by_stage))
        object.__setattr__(self, "max_stride_by_stage", tuple(int(v) for v in self.max_stride_by_stage))
        object.__setattr__(self, "start_q_noise", _tuple7(self.start_q_noise))
        object.__setattr__(self, "goal_q_noise", _tuple7(self.goal_q_noise))
        object.__setattr__(self, "_route_q", _load_route_q(self.route_path) if self.enabled else ())


@dataclass(frozen=True)
class ResetSample:
    initial_q: np.ndarray
    goal_q: np.ndarray
    goal_pose6: np.ndarray
    initial_dq: np.ndarray | None = None
    initial_prev_action: np.ndarray | None = None


def _load_route_q(path_str: str) -> tuple[tuple[float, ...], ...]:
    if not path_str:
        return ()
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Route reset path does not exist: {path}")
    payload = json.loads(path.read_text())
    raw_points = payload.get("route_q", payload.get("waypoints", payload if isinstance(payload, list) else []))
    points: list[tuple[float, ...]] = []
    for item in raw_points:
        q = item.get("q") if isinstance(item, dict) else item
        points.append(_tuple7(q))
    if len(points) < 2:
        raise ValueError(f"Route reset path must contain at least two q waypoints: {path}")
    return tuple(points)


def _load_handoff_states(
    path_str: str,
    *,
    max_position_error_m: float,
    max_orientation_error_rad: float,
    max_action_l2: float,
) -> tuple[HandoffResetState, ...]:
    if not path_str:
        return ()
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Handoff state buffer does not exist: {path}")
    payload = json.loads(path.read_text())
    raw_states = payload.get("states", payload if isinstance(payload, list) else [])
    states: list[HandoffResetState] = []
    for item in raw_states:
        pos_err = float(item.get("position_error_norm", 0.0))
        ori_err = float(item.get("orientation_error_norm", 0.0))
        action_l2 = float(item.get("action_l2", 0.0))
        if pos_err > max_position_error_m:
            continue
        if ori_err > max_orientation_error_rad:
            continue
        if action_l2 > max_action_l2:
            continue
        states.append(
            HandoffResetState(
                initial_q=item["initial_q"],
                goal_q=item["goal_q"],
                goal_pose6=item["goal_pose6"],
                initial_dq=item.get("initial_dq", [0.0] * 7),
                initial_prev_action=item.get("initial_prev_action", [0.0] * 7),
            )
        )
    return tuple(states)


def sample_approach_reset(
    *,
    rng: np.random.Generator,
    joint_specs: Sequence[JointSpec],
    curriculum_config: PointCurriculumConfig,
    stage_index: int,
    start_margin_fraction: float,
    goal_margin_fraction: float,
    route_reset_config: RouteResetConfig | None = None,
    workspace_stage_sampling: dict[str, object] | None = None,
) -> ResetSample:
    if route_reset_config is not None and route_reset_config.enabled and route_reset_config._route_q:
        return sample_route_approach_reset(
            rng=rng,
            joint_specs=joint_specs,
            route_reset_config=route_reset_config,
            stage_index=stage_index,
        )
    random_start_cfg = dict((workspace_stage_sampling or {}).get("random_start_pair_sampling", {}))
    if bool(random_start_cfg.get("enabled", False)) and curriculum_config.enabled and curriculum_config.stages:
        return sample_random_start_workspace_pair(
            rng=rng,
            joint_specs=joint_specs,
            curriculum_config=curriculum_config,
            stage_index=stage_index,
            config=random_start_cfg,
            fallback_start_margin_fraction=start_margin_fraction,
            fallback_goal_margin_fraction=goal_margin_fraction,
        )
    if curriculum_config.enabled and curriculum_config.stages:
        idx = _sample_workspace_stage_index(
            rng=rng,
            current_stage_index=stage_index,
            stage_count=len(curriculum_config.stages),
            config=workspace_stage_sampling,
        )
        stage = curriculum_config.stages[idx]
        initial_q = sample_stage_joint_target(rng, stage.start_q, stage.start_noise, joint_specs)
        goal_q = sample_stage_joint_target(rng, stage.goal_q, stage.goal_noise, joint_specs)
    else:
        initial_q = sample_joint_configuration(rng, joint_specs, margin_fraction=start_margin_fraction)
        goal_q = sample_joint_configuration(rng, joint_specs, margin_fraction=goal_margin_fraction)
    return ResetSample(initial_q=initial_q, goal_q=goal_q, goal_pose6=compute_ee_pose6(goal_q))


def sample_random_start_workspace_pair(
    *,
    rng: np.random.Generator,
    joint_specs: Sequence[JointSpec],
    curriculum_config: PointCurriculumConfig,
    stage_index: int,
    config: dict[str, object] | None,
    fallback_start_margin_fraction: float,
    fallback_goal_margin_fraction: float,
) -> ResetSample:
    """Sample a mixed-start goal-conditioned pair for workspace coverage.

    This keeps the observation/action schema unchanged: the policy still sees
    current q/dq plus goal pose error, while reset no longer always starts at
    home.  The sampler is intentionally conservative and stage-aware so it can
    be used in overnight fine-tuning without immediately destroying the known
    Stage 0-7 behavior.
    """

    cfg = dict(config or {})
    stages = curriculum_config.stages
    current = int(np.clip(stage_index, 0, len(stages) - 1))

    source = _sample_ratio_key(
        rng,
        {
            "home": float(cfg.get("home_start_ratio", 0.15)),
            "old_success": float(cfg.get("old_successful_start_ratio", 0.25)),
            "random_valid": float(cfg.get("random_valid_q_start_ratio", 0.25)),
            "frontier": float(cfg.get("frontier_pair_ratio", 0.20)),
            "failure_recovery": float(cfg.get("failure_recovery_start_ratio", 0.10)),
            "stress": float(cfg.get("stress_start_ratio", 0.05)),
        },
        default="old_success",
    )

    target_stage = _sample_target_stage_for_source(rng=rng, source=source, current=current, stage_count=len(stages), config=cfg)
    target_cfg = stages[target_stage]
    target_q = sample_stage_joint_target(rng, target_cfg.goal_q, target_cfg.goal_noise, joint_specs)

    if source == "home":
        start_stage = min(int(cfg.get("home_stage_index", 0)), len(stages) - 1)
        start_cfg = stages[start_stage]
        start_q = sample_stage_joint_target(rng, start_cfg.start_q, start_cfg.start_noise, joint_specs)
    elif source == "old_success":
        max_old = int(np.clip(cfg.get("old_success_max_stage_index", min(7, current)), 0, len(stages) - 1))
        old_idx = int(rng.integers(0, max_old + 1))
        old_stage = stages[old_idx]
        # Successful final states are approximated by old-stage goals.  This is
        # safer than hard-resetting to arbitrary q while still breaking the
        # home-start dependency.
        start_q = sample_stage_joint_target(rng, old_stage.goal_q, old_stage.goal_noise, joint_specs)
    elif source == "frontier":
        frontier_min = int(np.clip(cfg.get("frontier_min_stage_index", min(8, current)), 0, len(stages) - 1))
        frontier_max = int(np.clip(cfg.get("frontier_max_stage_index", current), frontier_min, len(stages) - 1))
        front_idx = int(rng.integers(frontier_min, frontier_max + 1))
        front_stage = stages[front_idx]
        start_q = sample_stage_joint_target(rng, front_stage.start_q, front_stage.start_noise, joint_specs)
    elif source == "failure_recovery":
        # Recovery starts near the target with extra q/dq/prev_action noise so
        # the policy learns to settle after drift instead of only approaching.
        recovery_noise = np.asarray(cfg.get("failure_recovery_q_noise", [0.04] * len(joint_specs)), dtype=float)
        start_q = clip_joint_configuration(target_q + rng.uniform(-recovery_noise, recovery_noise), joint_specs)
    elif source == "stress":
        margin = float(cfg.get("stress_start_margin_fraction", fallback_start_margin_fraction))
        start_q = sample_joint_configuration(rng, joint_specs, margin_fraction=margin)
    else:
        margin = float(cfg.get("random_valid_start_margin_fraction", fallback_start_margin_fraction))
        start_q = sample_joint_configuration(rng, joint_specs, margin_fraction=margin)

    dq_noise = np.asarray(cfg.get("initial_dq_noise", [0.0] * len(joint_specs)), dtype=float)
    prev_action_noise = np.asarray(cfg.get("initial_prev_action_noise", [0.0] * len(joint_specs)), dtype=float)
    initial_dq = rng.uniform(-dq_noise, dq_noise) if np.any(dq_noise > 0.0) else np.zeros(len(joint_specs), dtype=float)
    initial_prev_action = rng.uniform(-prev_action_noise, prev_action_noise) if np.any(prev_action_noise > 0.0) else np.zeros(len(joint_specs), dtype=float)

    min_q_l2 = float(cfg.get("min_pair_joint_l2", 0.0))
    if min_q_l2 > 0.0:
        # Try a few times to avoid degenerate same-start/same-target episodes.
        for _ in range(12):
            if float(np.linalg.norm(target_q - start_q)) >= min_q_l2:
                break
            target_stage = _sample_target_stage_for_source(rng=rng, source=source, current=current, stage_count=len(stages), config=cfg)
            target_cfg = stages[target_stage]
            target_q = sample_stage_joint_target(rng, target_cfg.goal_q, target_cfg.goal_noise, joint_specs)

    clipped_target_q = clip_joint_configuration(target_q, joint_specs)
    return ResetSample(
        initial_q=clip_joint_configuration(start_q, joint_specs),
        goal_q=clipped_target_q,
        goal_pose6=compute_ee_pose6(clipped_target_q),
        initial_dq=initial_dq,
        initial_prev_action=initial_prev_action,
    )


def _sample_ratio_key(rng: np.random.Generator, ratios: dict[str, float], *, default: str) -> str:
    clean = {k: max(float(v), 0.0) for k, v in ratios.items()}
    total = sum(clean.values())
    if total <= 0.0:
        return default
    draw = float(rng.random() * total)
    for key, value in clean.items():
        if draw <= value:
            return key
        draw -= value
    return default


def _sample_target_stage_for_source(
    *,
    rng: np.random.Generator,
    source: str,
    current: int,
    stage_count: int,
    config: dict[str, object],
) -> int:
    if source in {"home", "old_success"}:
        max_stage = int(np.clip(config.get("known_target_max_stage_index", min(7, current)), 0, stage_count - 1))
        return int(rng.integers(0, max_stage + 1))
    if source == "frontier":
        min_stage = int(np.clip(config.get("frontier_target_min_stage_index", min(8, current)), 0, stage_count - 1))
        max_stage = int(np.clip(config.get("frontier_target_max_stage_index", current), min_stage, stage_count - 1))
        return int(rng.integers(min_stage, max_stage + 1))
    if source == "stress":
        min_stage = int(np.clip(config.get("stress_target_min_stage_index", min(8, current)), 0, stage_count - 1))
        max_stage = int(np.clip(config.get("stress_target_max_stage_index", stage_count - 1), min_stage, stage_count - 1))
        return int(rng.integers(min_stage, max_stage + 1))
    max_stage = int(np.clip(config.get("mixed_target_max_stage_index", current), 0, stage_count - 1))
    return int(rng.integers(0, max_stage + 1))


def _sample_workspace_stage_index(
    *,
    rng: np.random.Generator,
    current_stage_index: int,
    stage_count: int,
    config: dict[str, object] | None,
) -> int:
    """Sample a curriculum stage with old-stage replay for workspace expansion.

    The active curriculum callback still controls the maximum/current expansion
    stage, while this mixer prevents catastrophic forgetting by replaying
    earlier workspace stages inside the same training batch.
    """

    current = int(np.clip(current_stage_index, 0, max(stage_count - 1, 0)))
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)) or current <= 0:
        return current

    current_ratio = max(float(cfg.get("current_stage_ratio", 0.50)), 0.0)
    previous_ratio = max(float(cfg.get("previous_stage_ratio", 0.25)), 0.0)
    old_ratio = max(float(cfg.get("old_workspace_replay_ratio", 0.20)), 0.0)
    # Failure replay is represented as additional old/previous sampling until a
    # target buffer is wired in; keep the ratio in config for auditability.
    failure_ratio = max(float(cfg.get("failure_replay_ratio", 0.05)), 0.0)
    total = current_ratio + previous_ratio + old_ratio + failure_ratio
    if total <= 0.0:
        return current

    draw = float(rng.random() * total)
    if draw < current_ratio:
        return current
    draw -= current_ratio

    if draw < previous_ratio and current > 0:
        low = max(int(cfg.get("previous_stage_min_index", 0)), 0)
        high = max(current - 1, low)
        return int(rng.integers(low, high + 1))
    draw -= previous_ratio

    old_max = int(cfg.get("old_workspace_max_stage_index", min(5, current)))
    old_max = int(np.clip(old_max, 0, min(stage_count - 1, current)))
    if draw < old_ratio and old_max >= 0:
        return int(rng.integers(0, old_max + 1))

    replay_max = max(min(old_max, current - 1), 0)
    return int(rng.integers(0, replay_max + 1)) if replay_max > 0 else current


def sample_route_approach_reset(
    *,
    rng: np.random.Generator,
    joint_specs: Sequence[JointSpec],
    route_reset_config: RouteResetConfig,
    stage_index: int,
) -> ResetSample:
    route_q = route_reset_config._route_q
    stage_idx = int(max(stage_index, 0))
    min_stride = route_reset_config.min_stride_by_stage[min(stage_idx, len(route_reset_config.min_stride_by_stage) - 1)]
    max_stride = route_reset_config.max_stride_by_stage[min(stage_idx, len(route_reset_config.max_stride_by_stage) - 1)]
    min_stride = max(int(min_stride), 1)
    max_stride = max(int(max_stride), min_stride)
    max_stride = min(max_stride, len(route_q) - 1)
    stride = int(rng.integers(min_stride, max_stride + 1))
    start_index = int(rng.integers(0, len(route_q) - stride))
    goal_index = start_index + stride
    if route_reset_config.reverse_probability > 0.0 and rng.random() < route_reset_config.reverse_probability:
        start_index, goal_index = goal_index, start_index

    initial_q = np.asarray(route_q[start_index], dtype=float)
    goal_q = np.asarray(route_q[goal_index], dtype=float)
    start_noise = np.asarray(route_reset_config.start_q_noise, dtype=float)
    goal_noise = np.asarray(route_reset_config.goal_q_noise, dtype=float)
    if np.any(start_noise > 0.0):
        initial_q = initial_q + rng.uniform(low=-start_noise, high=start_noise)
    if np.any(goal_noise > 0.0):
        goal_q = goal_q + rng.uniform(low=-goal_noise, high=goal_noise)
    initial_q = clip_joint_configuration(initial_q, joint_specs)
    goal_q = clip_joint_configuration(goal_q, joint_specs)
    return ResetSample(initial_q=initial_q, goal_q=goal_q, goal_pose6=compute_ee_pose6(goal_q))


def sample_dock_reset(
    *,
    rng: np.random.Generator,
    joint_specs: Sequence[JointSpec],
    dock_reset_config: DockResetConfig,
    curriculum_config: PointCurriculumConfig,
    stage_index: int,
) -> ResetSample:
    if (
        dock_reset_config.handoff_state_probability > 0.0
        and dock_reset_config._handoff_states
        and rng.random() < dock_reset_config.handoff_state_probability
    ):
        state = dock_reset_config._handoff_states[int(rng.integers(len(dock_reset_config._handoff_states)))]
        return ResetSample(
            initial_q=np.asarray(state.initial_q, dtype=float),
            goal_q=np.asarray(state.goal_q, dtype=float),
            goal_pose6=np.asarray(state.goal_pose6, dtype=float),
            initial_dq=np.asarray(state.initial_dq, dtype=float),
            initial_prev_action=np.asarray(state.initial_prev_action, dtype=float),
        )

    if curriculum_config.enabled and curriculum_config.stages:
        idx = int(np.clip(stage_index, 0, len(curriculum_config.stages) - 1))
        stage = curriculum_config.stages[idx]
        goal_q = sample_stage_joint_target(rng, stage.goal_q, stage.goal_noise, joint_specs)
    else:
        goal_q = sample_stage_joint_target(rng, dock_reset_config.goal_q, dock_reset_config.goal_noise, joint_specs)

    goal_pose6 = compute_ee_pose6(goal_q)
    if dock_reset_config.close_bucket_probability > 0.0 and rng.random() < dock_reset_config.close_bucket_probability:
        initial_q = _sample_close_bucket_initial_q(
            rng=rng,
            joint_specs=joint_specs,
            goal_q=goal_q,
            goal_pose6=goal_pose6,
            dock_reset_config=dock_reset_config,
        )
        return ResetSample(initial_q=initial_q, goal_q=goal_q, goal_pose6=goal_pose6)

    init_delta = rng.uniform(
        low=-np.asarray(dock_reset_config.init_q_noise, dtype=float),
        high=np.asarray(dock_reset_config.init_q_noise, dtype=float),
    )
    initial_q = clip_joint_configuration(goal_q + init_delta, joint_specs)
    return ResetSample(initial_q=initial_q, goal_q=goal_q, goal_pose6=goal_pose6)


def _sample_close_bucket_initial_q(
    *,
    rng: np.random.Generator,
    joint_specs: Sequence[JointSpec],
    goal_q: np.ndarray,
    goal_pose6: np.ndarray,
    dock_reset_config: DockResetConfig,
) -> np.ndarray:
    """Sample near-success but not-yet-success dock states for last-centimeter practice."""

    noise = np.asarray(dock_reset_config.close_init_q_noise, dtype=float)
    best_q: np.ndarray | None = None
    best_distance_to_bucket = float("inf")
    for _ in range(max(int(dock_reset_config.close_bucket_max_attempts), 1)):
        init_delta = rng.uniform(low=-noise, high=noise)
        candidate_q = clip_joint_configuration(goal_q + init_delta, joint_specs)
        candidate_pose6 = compute_ee_pose6(candidate_q)
        pos_err, ori_err = pose_error_components(candidate_pose6, goal_pose6)
        pos_norm = l2_norm(pos_err)
        ori_norm = l2_norm(ori_err)
        if (
            dock_reset_config.close_bucket_min_pos_error_m <= pos_norm <= dock_reset_config.close_bucket_max_pos_error_m
            and ori_norm >= dock_reset_config.close_bucket_min_ori_error_rad
            and ori_norm <= dock_reset_config.close_bucket_max_ori_error_rad
        ):
            return candidate_q

        if pos_norm < dock_reset_config.close_bucket_min_pos_error_m:
            bucket_distance = dock_reset_config.close_bucket_min_pos_error_m - pos_norm
        elif pos_norm > dock_reset_config.close_bucket_max_pos_error_m:
            bucket_distance = pos_norm - dock_reset_config.close_bucket_max_pos_error_m
        else:
            bucket_distance = max(
                dock_reset_config.close_bucket_min_ori_error_rad - ori_norm,
                ori_norm - dock_reset_config.close_bucket_max_ori_error_rad,
                0.0,
            )
        if bucket_distance < best_distance_to_bucket:
            best_q = candidate_q
            best_distance_to_bucket = float(bucket_distance)

    return best_q if best_q is not None else clip_joint_configuration(goal_q, joint_specs)
