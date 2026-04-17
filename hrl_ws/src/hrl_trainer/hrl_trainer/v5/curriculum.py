"""V5 WP2 curriculum scaffold (Stage A/B/C) with selector hook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class CurriculumStageConfig:
    stage_id: str
    episode_start: int
    episode_end: int | None = None
    seed_set: tuple[int, ...] = ()
    reward_config: dict[str, float] = field(default_factory=dict)

    def contains(self, episode_index: int) -> bool:
        if episode_index < self.episode_start:
            return False
        if self.episode_end is None:
            return True
        return episode_index <= self.episode_end


@dataclass(frozen=True)
class CurriculumConfig:
    stages: tuple[CurriculumStageConfig, ...]

    @classmethod
    def from_stage_mappings(cls, payload: Sequence[Mapping[str, object]]) -> "CurriculumConfig":
        stages: list[CurriculumStageConfig] = []
        for row in payload:
            stages.append(
                CurriculumStageConfig(
                    stage_id=str(row["stage_id"]),
                    episode_start=int(row["episode_start"]),
                    episode_end=int(row["episode_end"]) if row.get("episode_end") is not None else None,
                    seed_set=tuple(int(x) for x in row.get("seed_set", ())),
                    reward_config={k: float(v) for k, v in dict(row.get("reward_config", {})).items()},
                )
            )
        return cls(stages=tuple(stages))


SelectorHook = Callable[[int, CurriculumConfig], str | None]


class CurriculumSelector:
    """Stage selector with optional hook for custom scheduling logic."""

    def __init__(self, config: CurriculumConfig, *, selector_hook: SelectorHook | None = None):
        self._config = config
        self._selector_hook = selector_hook

    @property
    def config(self) -> CurriculumConfig:
        return self._config

    def select_stage(self, episode_index: int) -> CurriculumStageConfig:
        if self._selector_hook is not None:
            stage_id = self._selector_hook(int(episode_index), self._config)
            if stage_id is not None:
                for stage in self._config.stages:
                    if stage.stage_id == stage_id:
                        return stage
                raise KeyError(f"selector_hook returned unknown stage_id: {stage_id}")

        for stage in self._config.stages:
            if stage.contains(int(episode_index)):
                return stage
        raise IndexError(f"No curriculum stage configured for episode_index={episode_index}")


def default_stage_abc_config() -> CurriculumConfig:
    """Default Stage A/B/C scaffold (U-slot-first progression)."""

    return CurriculumConfig(
        stages=(
            CurriculumStageConfig(
                stage_id="A",
                episode_start=0,
                episode_end=999,
                seed_set=(101, 102, 103),
                reward_config={
                    "terminal_success_reward": 1.0,
                    "terminal_failure_penalty": -1.0,
                    "pbrs_gamma": 0.95,
                },
            ),
            CurriculumStageConfig(
                stage_id="B",
                episode_start=1000,
                episode_end=2999,
                seed_set=(201, 202, 203),
                reward_config={
                    "terminal_success_reward": 2.0,
                    "terminal_failure_penalty": -1.5,
                    "pbrs_gamma": 0.97,
                },
            ),
            CurriculumStageConfig(
                stage_id="C",
                episode_start=3000,
                episode_end=None,
                seed_set=(301, 302, 303),
                reward_config={
                    "terminal_success_reward": 3.0,
                    "terminal_failure_penalty": -2.0,
                    "pbrs_gamma": 0.99,
                },
            ),
        )
    )
