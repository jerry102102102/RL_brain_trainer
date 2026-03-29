"""Reward composition for V5.1 real SAC pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    w_progress: float = 1.0
    w_action_norm: float = -0.15
    w_jerk: float = -0.10
    w_intervention: float = -1.0
    w_clamp_projection: float = -0.35
    timeout_penalty: float = -2.5
    reset_fail_penalty: float = -3.0
    success_bonus: float = 3.0


@dataclass(frozen=True)
class RewardTerms:
    progress: float
    action: float
    jerk: float
    intervention: float
    clamp_or_projection: float
    timeout_or_reset: float
    success_bonus: float
    reward_total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "action": self.action,
            "jerk": self.jerk,
            "intervention": self.intervention,
            "clamp_or_projection": self.clamp_or_projection,
            "timeout_or_reset": self.timeout_or_reset,
            "success_bonus": self.success_bonus,
            "reward_total": self.reward_total,
        }


class RewardComposer:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        prev_error: float,
        curr_error: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        intervention: bool,
        clamp_or_projection: bool,
        done: bool,
        done_reason: str,
    ) -> RewardTerms:
        progress = self.config.w_progress * float(prev_error - curr_error)
        action_term = self.config.w_action_norm * float(np.linalg.norm(action))
        jerk = self.config.w_jerk * float(np.linalg.norm(action - prev_action))
        intervention_term = self.config.w_intervention if intervention else 0.0
        clamp_term = self.config.w_clamp_projection if clamp_or_projection else 0.0

        timeout_reset_term = 0.0
        success_bonus = 0.0
        if done:
            if done_reason in {"timeout", "reset_fail"}:
                timeout_reset_term = (
                    self.config.timeout_penalty if done_reason == "timeout" else self.config.reset_fail_penalty
                )
            elif done_reason == "success":
                success_bonus = self.config.success_bonus

        total = (
            progress
            + action_term
            + jerk
            + intervention_term
            + clamp_term
            + timeout_reset_term
            + success_bonus
        )

        return RewardTerms(
            progress=progress,
            action=action_term,
            jerk=jerk,
            intervention=intervention_term,
            clamp_or_projection=clamp_term,
            timeout_or_reset=timeout_reset_term,
            success_bonus=success_bonus,
            reward_total=float(total),
        )


class RewardTraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
