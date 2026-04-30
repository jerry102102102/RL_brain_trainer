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
class ResetSample:
    initial_q: np.ndarray
    goal_q: np.ndarray
    goal_pose6: np.ndarray
    initial_dq: np.ndarray | None = None
    initial_prev_action: np.ndarray | None = None


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
) -> ResetSample:
    if curriculum_config.enabled and curriculum_config.stages:
        idx = int(np.clip(stage_index, 0, len(curriculum_config.stages) - 1))
        stage = curriculum_config.stages[idx]
        initial_q = sample_stage_joint_target(rng, stage.start_q, stage.start_noise, joint_specs)
        goal_q = sample_stage_joint_target(rng, stage.goal_q, stage.goal_noise, joint_specs)
    else:
        initial_q = sample_joint_configuration(rng, joint_specs, margin_fraction=start_margin_fraction)
        goal_q = sample_joint_configuration(rng, joint_specs, margin_fraction=goal_margin_fraction)
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
