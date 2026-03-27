"""V5.1 formal acceptance gates (T8).

Provides P0/P1 gate checks with a stable JSON output schema that can block
promotion when quality/stability conditions are not met.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GATE_VERSION = "v5.1.gates.v2"


@dataclass(frozen=True)
class GateSpec:
    gate_version: str = GATE_VERSION
    max_reset_failures: int = 0
    min_execution_ratio: float = 1.0
    min_log_integrity_ratio: float = 1.0
    min_success_trend_delta: float = 0.0
    max_intervention_worsen: float = 0.0


@dataclass(frozen=True)
class GateCheckResult:
    name: str
    priority: str
    passed: bool
    reason: str
    metrics: dict[str, float]
    threshold: dict[str, float | str]


@dataclass(frozen=True)
class GateResult:
    gate_version: str
    run_id: str
    each_gate: list[GateCheckResult]
    overall_decision: str


DEFAULT_GATE = GateSpec()


class GateEvaluator:
    def __init__(self, spec: GateSpec | None = None) -> None:
        self.spec = spec or DEFAULT_GATE

    @staticmethod
    def _check(
        name: str,
        priority: str,
        passed: bool,
        reason_ok: str,
        reason_fail: str,
        metrics: dict[str, float],
        threshold: dict[str, float | str],
    ) -> GateCheckResult:
        return GateCheckResult(
            name=name,
            priority=priority,
            passed=bool(passed),
            reason=reason_ok if passed else reason_fail,
            metrics=metrics,
            threshold=threshold,
        )

    def evaluate(self, run_id: str, metrics: dict[str, Any]) -> GateResult:
        episodes_requested = max(1, int(metrics.get("episodes_requested", 0)))
        episodes_completed = int(metrics.get("episodes_completed", 0))
        execution_ratio = float(episodes_completed) / float(episodes_requested)

        reset_failures = int(metrics.get("reset_failures", 0))
        log_expected = max(1, int(metrics.get("log_lines_expected_per_layer", 0)))
        l1_lines = int(metrics.get("l1_log_lines", 0))
        l2_lines = int(metrics.get("l2_log_lines", 0))
        l3_lines = int(metrics.get("l3_log_lines", 0))
        min_layer_lines = min(l1_lines, l2_lines, l3_lines)
        log_integrity_ratio = float(min_layer_lines) / float(log_expected)

        success_first = float(metrics.get("success_rate_first", 0.0))
        success_last = float(metrics.get("success_rate_last", 0.0))
        success_delta = success_last - success_first

        intervention_first = float(metrics.get("intervention_rate_first", 1.0))
        intervention_last = float(metrics.get("intervention_rate_last", 1.0))
        intervention_delta = intervention_last - intervention_first

        checks = [
            self._check(
                name="P0.reset_fail_fast",
                priority="P0",
                passed=reset_failures <= self.spec.max_reset_failures,
                reason_ok="No reset failures detected",
                reason_fail=f"reset_failures={reset_failures} exceeds {self.spec.max_reset_failures}",
                metrics={"reset_failures": float(reset_failures)},
                threshold={"op": "<=", "value": float(self.spec.max_reset_failures)},
            ),
            self._check(
                name="P0.execution_complete",
                priority="P0",
                passed=execution_ratio >= self.spec.min_execution_ratio,
                reason_ok="All requested episodes completed",
                reason_fail=(
                    f"execution_ratio={execution_ratio:.3f} below {self.spec.min_execution_ratio:.3f}"
                ),
                metrics={
                    "episodes_requested": float(episodes_requested),
                    "episodes_completed": float(episodes_completed),
                    "execution_ratio": execution_ratio,
                },
                threshold={"op": ">=", "value": float(self.spec.min_execution_ratio)},
            ),
            self._check(
                name="P0.log_integrity",
                priority="P0",
                passed=log_integrity_ratio >= self.spec.min_log_integrity_ratio,
                reason_ok="Layer logs are complete",
                reason_fail=(
                    f"log_integrity_ratio={log_integrity_ratio:.3f} below {self.spec.min_log_integrity_ratio:.3f}"
                ),
                metrics={
                    "l1_log_lines": float(l1_lines),
                    "l2_log_lines": float(l2_lines),
                    "l3_log_lines": float(l3_lines),
                    "expected_per_layer": float(log_expected),
                    "log_integrity_ratio": log_integrity_ratio,
                },
                threshold={"op": ">=", "value": float(self.spec.min_log_integrity_ratio)},
            ),
            self._check(
                name="P1.success_trend",
                priority="P1",
                passed=success_delta >= self.spec.min_success_trend_delta,
                reason_ok="Success trend is non-degrading",
                reason_fail=(
                    f"success_delta={success_delta:.3f} below {self.spec.min_success_trend_delta:.3f}"
                ),
                metrics={
                    "success_rate_first": success_first,
                    "success_rate_last": success_last,
                    "success_delta": success_delta,
                },
                threshold={"op": ">=", "value": float(self.spec.min_success_trend_delta)},
            ),
            self._check(
                name="P1.intervention_non_worsening",
                priority="P1",
                passed=intervention_delta <= self.spec.max_intervention_worsen,
                reason_ok="Intervention rate is non-worsening",
                reason_fail=(
                    f"intervention_delta={intervention_delta:.3f} exceeds {self.spec.max_intervention_worsen:.3f}"
                ),
                metrics={
                    "intervention_rate_first": intervention_first,
                    "intervention_rate_last": intervention_last,
                    "intervention_delta": intervention_delta,
                },
                threshold={"op": "<=", "value": float(self.spec.max_intervention_worsen)},
            ),
        ]

        overall_decision = "GO" if all(check.passed for check in checks) else "HOLD"
        return GateResult(
            gate_version=self.spec.gate_version,
            run_id=run_id,
            each_gate=checks,
            overall_decision=overall_decision,
        )


def write_gate_report(path: Path, result: GateResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
