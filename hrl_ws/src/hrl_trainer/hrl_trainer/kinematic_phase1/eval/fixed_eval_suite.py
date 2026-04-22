"""Fixed deterministic evaluation suite for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from ..envs.arm_kinematic_env import Phase1EnvConfig
from ..envs.curriculum import sample_stage_joint_target
from ..envs.reset_samplers import sample_dock_reset
from ..kinematics.fk_interface import compute_ee_pose6
from ..kinematics.joint_limits import JointSpec, sample_joint_configuration


@dataclass(frozen=True)
class EvalEpisodeSpec:
    episode_id: int
    initial_q: list[float]
    goal_q: list[float]
    goal_pose6: list[float]

    def reset_options(self) -> dict[str, list[float]]:
        return {
            "initial_q": self.initial_q,
            "goal_q": self.goal_q,
            "goal_pose6": self.goal_pose6,
        }


def build_fixed_eval_suite(
    *,
    seed: int,
    n_episodes: int,
    joint_specs: Sequence[JointSpec],
    start_margin_fraction: float = 0.20,
    goal_margin_fraction: float = 0.10,
) -> list[EvalEpisodeSpec]:
    rng = np.random.default_rng(seed)
    episodes: list[EvalEpisodeSpec] = []
    for idx in range(n_episodes):
        initial_q = sample_joint_configuration(rng, joint_specs, margin_fraction=start_margin_fraction)
        goal_q = sample_joint_configuration(rng, joint_specs, margin_fraction=goal_margin_fraction)
        goal_pose6 = compute_ee_pose6(goal_q)
        episodes.append(
            EvalEpisodeSpec(
                episode_id=idx,
                initial_q=initial_q.tolist(),
                goal_q=goal_q.tolist(),
                goal_pose6=goal_pose6.tolist(),
            )
        )
    return episodes


def suite_to_jsonable(suite: Sequence[EvalEpisodeSpec]) -> list[dict[str, object]]:
    return [asdict(item) for item in suite]


def default_eval_suite(config: Phase1EnvConfig, *, seed: int = 700001, n_episodes: int = 10) -> list[EvalEpisodeSpec]:
    return build_fixed_eval_suite(
        seed=seed,
        n_episodes=n_episodes,
        joint_specs=config.joint_specs,
        start_margin_fraction=config.start_sample_margin_fraction,
        goal_margin_fraction=config.goal_sample_margin_fraction,
    )


def build_curriculum_local_eval_suite(
    config: Phase1EnvConfig,
    *,
    seed: int = 700001,
    stage_index: int = 0,
    n_episodes: int = 10,
) -> list[EvalEpisodeSpec]:
    if not config.curriculum_config.enabled:
        raise ValueError("Curriculum-local eval requires curriculum to be enabled")
    if not config.curriculum_config.stages:
        raise ValueError("No curriculum stages are defined")
    idx = int(np.clip(stage_index, 0, len(config.curriculum_config.stages) - 1))
    stage = config.curriculum_config.stages[idx]
    rng = np.random.default_rng(seed)
    episodes: list[EvalEpisodeSpec] = []
    for episode_id in range(n_episodes):
        initial_q = sample_stage_joint_target(rng, stage.start_q, stage.start_noise, config.joint_specs)
        goal_q = sample_stage_joint_target(rng, stage.goal_q, stage.goal_noise, config.joint_specs)
        goal_pose6 = compute_ee_pose6(goal_q)
        episodes.append(
            EvalEpisodeSpec(
                episode_id=episode_id,
                initial_q=initial_q.tolist(),
                goal_q=goal_q.tolist(),
                goal_pose6=goal_pose6.tolist(),
            )
        )
    return episodes


def build_dock_eval_suite(
    config: Phase1EnvConfig,
    *,
    seed: int = 700001,
    n_episodes: int = 10,
) -> list[EvalEpisodeSpec]:
    rng = np.random.default_rng(seed)
    episodes: list[EvalEpisodeSpec] = []
    for episode_id in range(n_episodes):
        sample = sample_dock_reset(
            rng=rng,
            joint_specs=config.joint_specs,
            dock_reset_config=config.dock_reset_config,
            curriculum_config=config.curriculum_config,
            stage_index=config.curriculum_config.stages and 0 or 0,
        )
        episodes.append(
            EvalEpisodeSpec(
                episode_id=episode_id,
                initial_q=sample.initial_q.tolist(),
                goal_q=sample.goal_q.tolist(),
                goal_pose6=sample.goal_pose6.tolist(),
            )
        )
    return episodes
