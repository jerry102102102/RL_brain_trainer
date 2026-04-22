"""Reset samplers for approach and dock Phase 1B modes."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class DockResetConfig:
    goal_q: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    goal_noise: tuple[float, ...] = (0.01, 0.03, 0.04, 0.03, 0.02, 0.02, 0.01)
    init_q_noise: tuple[float, ...] = (0.01, 0.02, 0.03, 0.02, 0.015, 0.015, 0.01)
    close_bucket_probability: float = 0.0
    close_init_q_noise: tuple[float, ...] = (0.006, 0.012, 0.018, 0.012, 0.009, 0.009, 0.006)
    close_bucket_min_pos_error_m: float = 0.005
    close_bucket_max_pos_error_m: float = 0.020
    close_bucket_max_ori_error_rad: float = 0.12
    close_bucket_max_attempts: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_q", _tuple7(self.goal_q))
        object.__setattr__(self, "goal_noise", _tuple7(self.goal_noise))
        object.__setattr__(self, "init_q_noise", _tuple7(self.init_q_noise))
        object.__setattr__(self, "close_init_q_noise", _tuple7(self.close_init_q_noise))


@dataclass(frozen=True)
class ResetSample:
    initial_q: np.ndarray
    goal_q: np.ndarray
    goal_pose6: np.ndarray


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
            and ori_norm <= dock_reset_config.close_bucket_max_ori_error_rad
        ):
            return candidate_q

        if pos_norm < dock_reset_config.close_bucket_min_pos_error_m:
            bucket_distance = dock_reset_config.close_bucket_min_pos_error_m - pos_norm
        elif pos_norm > dock_reset_config.close_bucket_max_pos_error_m:
            bucket_distance = pos_norm - dock_reset_config.close_bucket_max_pos_error_m
        else:
            bucket_distance = max(ori_norm - dock_reset_config.close_bucket_max_ori_error_rad, 0.0)
        if bucket_distance < best_distance_to_bucket:
            best_q = candidate_q
            best_distance_to_bucket = float(bucket_distance)

    return best_q if best_q is not None else clip_joint_configuration(goal_q, joint_specs)
