"""V5.1 acceptance gates (T5).

Minimal executable gate checks for end-to-end promotion criteria.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateSpec:
    name: str
    min_episodes: int
    min_success_rate: float
    max_intervention_rate: float


@dataclass(frozen=True)
class GateResult:
    passed: bool
    score: float
    reasons: list[str]
    metrics: dict[str, float]


DEFAULT_GATE = GateSpec(
    name="v5_1_min_exec_gate",
    min_episodes=3,
    min_success_rate=0.70,
    max_intervention_rate=0.30,
)


class GateEvaluator:
    def __init__(self, spec: GateSpec | None = None) -> None:
        self.spec = spec or DEFAULT_GATE

    def evaluate(self, metrics: dict[str, Any]) -> GateResult:
        episodes = int(metrics.get("episodes", 0))
        success_rate = float(metrics.get("success_rate", 0.0))
        intervention_rate = float(metrics.get("intervention_rate", 1.0))

        reasons: list[str] = []
        if episodes < self.spec.min_episodes:
            reasons.append(f"episodes<{self.spec.min_episodes}")
        if success_rate < self.spec.min_success_rate:
            reasons.append(f"success_rate<{self.spec.min_success_rate:.2f}")
        if intervention_rate > self.spec.max_intervention_rate:
            reasons.append(f"intervention_rate>{self.spec.max_intervention_rate:.2f}")

        score = max(0.0, 1.0 - (len(reasons) / 3.0))
        return GateResult(
            passed=len(reasons) == 0,
            score=score,
            reasons=reasons,
            metrics={
                "episodes": float(episodes),
                "success_rate": success_rate,
                "intervention_rate": intervention_rate,
            },
        )


def write_gate_report(path: Path, spec: GateSpec, result: GateResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": asdict(spec),
        "result": asdict(result),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
