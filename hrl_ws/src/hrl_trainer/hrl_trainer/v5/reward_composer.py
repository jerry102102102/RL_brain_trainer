"""V5 WP1.5 reward composer scaffold with per-step breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RewardTermWeights:
    progress: float = 1.0
    safety: float = 1.0
    smoothness: float = 0.2
    coverage: float = 0.3
    subgoal: float = 1.5


@dataclass(frozen=True)
class RewardTermInput:
    progress: float
    safety: float
    smoothness: float
    coverage: float
    subgoal: float


@dataclass(frozen=True)
class RewardStepBreakdown:
    step_index: int
    raw_terms: dict[str, float]
    weighted_terms: dict[str, float]
    total_reward: float
    terminal: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeRewardBreakdown:
    steps: tuple[RewardStepBreakdown, ...]
    term_totals: dict[str, float]
    total_reward: float
    terminal_reason: str | None = None


class RewardComposer:
    """Compose weighted rewards while preserving per-step term accounting."""

    def __init__(self, weights: RewardTermWeights | None = None):
        self.weights = weights or RewardTermWeights()

    def compose_step(
        self,
        step_index: int,
        terms: RewardTermInput | Mapping[str, float],
        *,
        terminal: bool = False,
        notes: Sequence[str] = (),
    ) -> RewardStepBreakdown:
        if isinstance(terms, RewardTermInput):
            raw_terms = {
                "progress": float(terms.progress),
                "safety": float(terms.safety),
                "smoothness": float(terms.smoothness),
                "coverage": float(terms.coverage),
                "subgoal": float(terms.subgoal),
            }
        else:
            raw_terms = {
                "progress": float(terms.get("progress", 0.0)),
                "safety": float(terms.get("safety", 0.0)),
                "smoothness": float(terms.get("smoothness", 0.0)),
                "coverage": float(terms.get("coverage", 0.0)),
                "subgoal": float(terms.get("subgoal", 0.0)),
            }

        weighted_terms = {
            "progress": self.weights.progress * raw_terms["progress"],
            "safety": self.weights.safety * raw_terms["safety"],
            "smoothness": self.weights.smoothness * raw_terms["smoothness"],
            "coverage": self.weights.coverage * raw_terms["coverage"],
            "subgoal": self.weights.subgoal * raw_terms["subgoal"],
        }
        total_reward = sum(weighted_terms.values())
        return RewardStepBreakdown(
            step_index=int(step_index),
            raw_terms=raw_terms,
            weighted_terms=weighted_terms,
            total_reward=float(total_reward),
            terminal=bool(terminal),
            notes=tuple(str(note) for note in notes),
        )

    def compose_episode(
        self,
        steps: Sequence[RewardStepBreakdown],
        *,
        terminal_reason: str | None = None,
    ) -> EpisodeRewardBreakdown:
        term_totals = {
            "progress": 0.0,
            "safety": 0.0,
            "smoothness": 0.0,
            "coverage": 0.0,
            "subgoal": 0.0,
        }
        for step in steps:
            for name in term_totals:
                term_totals[name] += float(step.weighted_terms.get(name, 0.0))
        total_reward = sum(step.total_reward for step in steps)
        return EpisodeRewardBreakdown(
            steps=tuple(steps),
            term_totals=term_totals,
            total_reward=float(total_reward),
            terminal_reason=terminal_reason,
        )


def make_zero_step(step_index: int) -> RewardStepBreakdown:
    composer = RewardComposer()
    return composer.compose_step(step_index, RewardTermInput(0.0, 0.0, 0.0, 0.0, 0.0))
