"""V5 reward composer with backward-compatible API + v2 components.

Reward Composer v2 components:
- sparse terminal reward
- PBRS delta
- safety penalty
- smoothness penalty

Legacy fields (progress/safety/smoothness/coverage/subgoal) are still accepted so L1/L3
contracts can stay unchanged while WP2 adopts the new config-driven semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RewardTermWeights:
    """Backward-compatible weights for legacy and v2-mapped terms."""

    progress: float = 1.0
    safety: float = 1.0
    smoothness: float = 0.2
    coverage: float = 0.3
    subgoal: float = 1.5


@dataclass(frozen=True)
class RewardComposerConfig:
    """Config-driven v2 controls.

    Notes:
    - `terminal_*` default to 0.0 to preserve old behavior unless explicitly enabled.
    - PBRS delta uses `gamma * phi_next - phi_current` when both potentials are provided;
      otherwise it falls back to legacy `progress`.
    """

    terminal_success_reward: float = 0.0
    terminal_failure_penalty: float = 0.0
    pbrs_gamma: float = 0.99

    @classmethod
    def from_mapping(cls, payload: Mapping[str, float] | None) -> "RewardComposerConfig":
        if payload is None:
            return cls()
        return cls(
            terminal_success_reward=float(payload.get("terminal_success_reward", 0.0)),
            terminal_failure_penalty=float(payload.get("terminal_failure_penalty", 0.0)),
            pbrs_gamma=float(payload.get("pbrs_gamma", 0.99)),
        )


@dataclass(frozen=True)
class RewardTermInput:
    # Legacy-compatible inputs
    progress: float = 0.0
    safety: float = 0.0
    smoothness: float = 0.0
    coverage: float = 0.0
    subgoal: float = 0.0

    # v2 inputs
    potential_current: float | None = None
    potential_next: float | None = None
    safety_violation: float | None = None
    action_delta: float | None = None
    terminal_success: bool | None = None


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

    def __init__(
        self,
        weights: RewardTermWeights | None = None,
        config: RewardComposerConfig | Mapping[str, float] | None = None,
    ):
        self.weights = weights or RewardTermWeights()
        self.config = config if isinstance(config, RewardComposerConfig) else RewardComposerConfig.from_mapping(config)

    def _coerce_input(self, terms: RewardTermInput | Mapping[str, float]) -> RewardTermInput:
        if isinstance(terms, RewardTermInput):
            return terms
        return RewardTermInput(
            progress=float(terms.get("progress", 0.0)),
            safety=float(terms.get("safety", 0.0)),
            smoothness=float(terms.get("smoothness", 0.0)),
            coverage=float(terms.get("coverage", 0.0)),
            subgoal=float(terms.get("subgoal", 0.0)),
            potential_current=float(terms["potential_current"]) if "potential_current" in terms else None,
            potential_next=float(terms["potential_next"]) if "potential_next" in terms else None,
            safety_violation=float(terms["safety_violation"]) if "safety_violation" in terms else None,
            action_delta=float(terms["action_delta"]) if "action_delta" in terms else None,
            terminal_success=bool(terms["terminal_success"]) if "terminal_success" in terms else None,
        )

    def compose_step(
        self,
        step_index: int,
        terms: RewardTermInput | Mapping[str, float],
        *,
        terminal: bool = False,
        notes: Sequence[str] = (),
    ) -> RewardStepBreakdown:
        inp = self._coerce_input(terms)

        # v2: PBRS delta, with legacy fallback to progress term.
        if inp.potential_current is not None and inp.potential_next is not None:
            pbrs_delta = self.config.pbrs_gamma * float(inp.potential_next) - float(inp.potential_current)
        else:
            pbrs_delta = float(inp.progress)

        # v2: penalty terms, with legacy fallback to provided signed values.
        safety_penalty = -abs(float(inp.safety_violation)) if inp.safety_violation is not None else float(inp.safety)
        smoothness_penalty = -abs(float(inp.action_delta)) if inp.action_delta is not None else float(inp.smoothness)

        # v2: sparse terminal reward.
        sparse_terminal = 0.0
        if terminal:
            terminal_success = bool(inp.terminal_success) if inp.terminal_success is not None else ("success" in " ".join(notes).lower())
            sparse_terminal = (
                float(self.config.terminal_success_reward)
                if terminal_success
                else float(self.config.terminal_failure_penalty)
            )

        raw_terms = {
            "sparse_terminal": sparse_terminal,
            "pbrs_delta": pbrs_delta,
            "safety_penalty": safety_penalty,
            "smoothness_penalty": smoothness_penalty,
            "coverage": float(inp.coverage),
            "subgoal": float(inp.subgoal),
        }

        weighted_terms = {
            "sparse_terminal": raw_terms["sparse_terminal"],
            "pbrs_delta": self.weights.progress * raw_terms["pbrs_delta"],
            "safety_penalty": self.weights.safety * raw_terms["safety_penalty"],
            "smoothness_penalty": self.weights.smoothness * raw_terms["smoothness_penalty"],
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
            "sparse_terminal": 0.0,
            "pbrs_delta": 0.0,
            "safety_penalty": 0.0,
            "smoothness_penalty": 0.0,
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
    return composer.compose_step(step_index, RewardTermInput())
