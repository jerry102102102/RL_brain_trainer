"""Simple region-based curriculum for Phase 1."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..kinematics.joint_limits import JointSpec, clip_joint_configuration


def _tuple7(values: Iterable[float]) -> tuple[float, ...]:
    data = tuple(float(v) for v in values)
    if len(data) != 7:
        raise ValueError("Phase 1 curriculum stages require 7-joint vectors")
    return data


@dataclass(frozen=True)
class CurriculumStageConfig:
    name: str
    start_q: tuple[float, ...]
    goal_q: tuple[float, ...]
    start_noise: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    goal_noise: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_q", _tuple7(self.start_q))
        object.__setattr__(self, "goal_q", _tuple7(self.goal_q))
        object.__setattr__(self, "start_noise", _tuple7(self.start_noise))
        object.__setattr__(self, "goal_noise", _tuple7(self.goal_noise))


def default_point_curriculum_stages() -> tuple[CurriculumStageConfig, ...]:
    zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        CurriculumStageConfig(
            name="region_small",
            start_q=zero,
            goal_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            goal_noise=(0.01, 0.03, 0.04, 0.03, 0.02, 0.02, 0.01),
        ),
        CurriculumStageConfig(
            name="region_medium",
            start_q=zero,
            goal_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            goal_noise=(0.02, 0.06, 0.08, 0.06, 0.04, 0.04, 0.03),
        ),
        CurriculumStageConfig(
            name="region_medium_wide",
            start_q=zero,
            goal_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            goal_noise=(0.03, 0.09, 0.12, 0.09, 0.06, 0.05, 0.04),
        ),
        CurriculumStageConfig(
            name="region_large",
            start_q=zero,
            start_noise=(0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
            goal_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            goal_noise=(0.04, 0.12, 0.16, 0.12, 0.08, 0.06, 0.05),
        ),
        CurriculumStageConfig(
            name="region_large_offset",
            start_q=zero,
            start_noise=(0.00, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
            goal_q=(0.03, -0.04, 0.05, -0.03, 0.02, -0.01, 0.01),
            goal_noise=(0.05, 0.14, 0.18, 0.14, 0.09, 0.07, 0.06),
        ),
        CurriculumStageConfig(
            name="region_wide_local_random",
            start_q=zero,
            start_noise=(0.00, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
            goal_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            goal_noise=(0.06, 0.18, 0.22, 0.16, 0.10, 0.08, 0.07),
        ),
    )


@dataclass(frozen=True)
class PointCurriculumConfig:
    enabled: bool = True
    success_rate_threshold: float = 0.80
    window_episodes: int = 20
    min_episodes_per_stage: int = 30
    stages: tuple[CurriculumStageConfig, ...] = field(default_factory=default_point_curriculum_stages)


def sample_stage_joint_target(
    rng: np.random.Generator,
    base_q: Sequence[float],
    noise_q: Sequence[float],
    joint_specs: Sequence[JointSpec],
) -> np.ndarray:
    base = np.asarray(base_q, dtype=float)
    noise = np.asarray(noise_q, dtype=float)
    if np.any(noise > 0.0):
        delta = rng.uniform(low=-noise, high=noise)
        base = base + delta
    return clip_joint_configuration(base, joint_specs)


class PointCurriculumTracker:
    """Track recent success and promote stages when stable success is reached."""

    def __init__(self, config: PointCurriculumConfig) -> None:
        self.config = config
        self.stage_index = 0
        self.stage_episode_count = 0
        self.recent_successes: deque[int] = deque(maxlen=max(int(config.window_episodes), 1))
        self.history: list[dict[str, float | int | str]] = []

    @property
    def max_stage_index(self) -> int:
        return max(len(self.config.stages) - 1, 0)

    def record_episode(self, *, success: bool) -> bool:
        self.stage_episode_count += 1
        self.recent_successes.append(1 if success else 0)
        if self.stage_index >= self.max_stage_index:
            return False
        if self.stage_episode_count < self.config.min_episodes_per_stage:
            return False
        if len(self.recent_successes) < self.config.window_episodes:
            return False
        success_rate = float(sum(self.recent_successes)) / float(len(self.recent_successes))
        if success_rate < self.config.success_rate_threshold:
            return False

        prev_stage = self.stage_index
        self.stage_index += 1
        self.stage_episode_count = 0
        self.recent_successes.clear()
        self.history.append(
            {
                "from_stage_index": prev_stage,
                "to_stage_index": self.stage_index,
                "from_stage_name": self.config.stages[prev_stage].name,
                "to_stage_name": self.config.stages[self.stage_index].name,
                "trigger_success_rate": success_rate,
            }
        )
        return True

    def snapshot(self) -> dict[str, object]:
        success_rate = float(sum(self.recent_successes)) / float(len(self.recent_successes)) if self.recent_successes else 0.0
        return {
            "stage_index": self.stage_index,
            "stage_name": self.config.stages[self.stage_index].name,
            "stage_episode_count": self.stage_episode_count,
            "recent_success_rate": success_rate,
            "history": list(self.history),
        }
