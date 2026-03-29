"""V5.1 curriculum scheduler (T4).

Implements a minimal but deterministic S0/S1/S2 progression controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    name: str
    min_episodes: int
    promote_success_rate: float
    reward_scale: float
    step_budget: int
    action_limit: float = 0.05
    controlled_dofs: int = 6


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    stage_name: str
    success_rate: float
    promoted: bool


@dataclass
class CurriculumState:
    stage_index: int = 0
    episodes_in_stage: int = 0
    total_episodes: int = 0


DEFAULT_STAGES: tuple[StageSpec, ...] = (
    StageSpec(name="S0", min_episodes=2, promote_success_rate=0.60, reward_scale=0.5, step_budget=32, action_limit=0.05),
    StageSpec(name="S1", min_episodes=2, promote_success_rate=0.75, reward_scale=0.8, step_budget=48, action_limit=0.08),
    StageSpec(name="S2", min_episodes=2, promote_success_rate=0.90, reward_scale=1.0, step_budget=64, action_limit=0.10),
)

S0_B_STAGES: tuple[StageSpec, ...] = (
    StageSpec(name="S0_B", min_episodes=2, promote_success_rate=0.60, reward_scale=0.5, step_budget=32, action_limit=0.15),
    StageSpec(name="S1", min_episodes=2, promote_success_rate=0.75, reward_scale=0.8, step_budget=48, action_limit=0.10),
    StageSpec(name="S2", min_episodes=2, promote_success_rate=0.90, reward_scale=1.0, step_budget=64, action_limit=0.10),
)


STAGE_PROFILES: dict[str, tuple[StageSpec, ...]] = {
    "default": DEFAULT_STAGES,
    "s0_b": S0_B_STAGES,
}


def resolve_stages(profile: str = "default") -> tuple[StageSpec, ...]:
    key = str(profile).strip().lower()
    if key not in STAGE_PROFILES:
        raise ValueError(f"unknown curriculum profile: {profile}")
    return STAGE_PROFILES[key]


class CurriculumManager:
    def __init__(self, stages: tuple[StageSpec, ...] | None = None) -> None:
        self.stages: tuple[StageSpec, ...] = stages or DEFAULT_STAGES
        if len(self.stages) < 1:
            raise ValueError("curriculum requires at least one stage")

        self.state = CurriculumState()
        self.history: list[EpisodeRecord] = []

    @property
    def current_stage(self) -> StageSpec:
        return self.stages[self.state.stage_index]

    @property
    def is_terminal(self) -> bool:
        return self.state.stage_index == len(self.stages) - 1

    def record_episode(self, success_rate: float) -> EpisodeRecord:
        if not (0.0 <= float(success_rate) <= 1.0):
            raise ValueError("success_rate must be in [0, 1]")

        stage = self.current_stage
        self.state.total_episodes += 1
        self.state.episodes_in_stage += 1

        promote = (
            (not self.is_terminal)
            and self.state.episodes_in_stage >= stage.min_episodes
            and float(success_rate) >= stage.promote_success_rate
        )

        record = EpisodeRecord(
            episode_index=self.state.total_episodes - 1,
            stage_name=stage.name,
            success_rate=float(success_rate),
            promoted=promote,
        )
        self.history.append(record)

        if promote:
            self.state.stage_index += 1
            self.state.episodes_in_stage = 0

        return record

    def to_artifact(self) -> dict[str, Any]:
        return {
            "state": asdict(self.state),
            "current_stage": asdict(self.current_stage),
            "history": [asdict(r) for r in self.history],
            "stages": [asdict(s) for s in self.stages],
        }
