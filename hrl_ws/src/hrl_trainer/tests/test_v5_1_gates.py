from __future__ import annotations

import json

from hrl_trainer.v5_1.gates import DEFAULT_GATE, GateEvaluator, write_gate_report


def _base_metrics() -> dict[str, float | int]:
    return {
        "episodes_requested": 4,
        "episodes_completed": 4,
        "reset_failures": 0,
        "log_lines_expected_per_layer": 12,
        "l1_log_lines": 12,
        "l2_log_lines": 12,
        "l3_log_lines": 12,
        "success_rate": 0.70,
        "success_rate_first": 0.55,
        "success_rate_last": 0.80,
        "intervention_rate_first": 1.0,
        "intervention_rate_last": 0.0,
    }


def test_gate_evaluator_pass_case() -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    result = evaluator.evaluate(run_id="r_pass", metrics=_base_metrics())

    assert result.gate_version == DEFAULT_GATE.gate_version
    assert result.run_id == "r_pass"
    assert result.overall_decision == "GO"
    assert all(g.passed for g in result.each_gate)


def test_gate_evaluator_fail_case_reset_fail_fast() -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    metrics = _base_metrics()
    metrics["reset_failures"] = 1

    result = evaluator.evaluate(run_id="r_fail_reset", metrics=metrics)

    assert result.overall_decision == "HOLD"
    gate = next(g for g in result.each_gate if g.name == "P0.reset_fail_fast")
    assert gate.passed is False
    assert "reset_failures" in gate.reason


def test_gate_evaluator_fail_case_learning_quality() -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    metrics = _base_metrics()
    metrics["success_rate_first"] = 0.85
    metrics["success_rate_last"] = 0.70
    metrics["intervention_rate_first"] = 0.0
    metrics["intervention_rate_last"] = 1.0

    result = evaluator.evaluate(run_id="r_fail_quality", metrics=metrics)

    assert result.overall_decision == "HOLD"
    trend = next(g for g in result.each_gate if g.name == "P1.success_trend")
    iv = next(g for g in result.each_gate if g.name == "P1.intervention_non_worsening")
    assert trend.passed is False
    assert iv.passed is False


def test_gate_evaluator_fail_case_success_rate_zero_floor() -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    metrics = _base_metrics()
    metrics["success_rate"] = 0.0
    metrics["success_rate_first"] = 0.0
    metrics["success_rate_last"] = 0.0

    result = evaluator.evaluate(run_id="r_fail_success_zero", metrics=metrics)

    assert result.overall_decision == "HOLD"
    floor_gate = next(g for g in result.each_gate if g.name == "P1.success_rate_floor")
    assert floor_gate.passed is False
    assert "success_rate=0.000" in floor_gate.reason


def test_gate_report_schema_contains_required_fields(tmp_path) -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    result = evaluator.evaluate(run_id="schema_case", metrics=_base_metrics())

    path = write_gate_report(tmp_path / "gate_result.json", result)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["gate_version"] == DEFAULT_GATE.gate_version
    assert payload["run_id"] == "schema_case"
    assert payload["overall_decision"] in {"GO", "HOLD"}
    assert isinstance(payload["each_gate"], list)
    assert payload["each_gate"]

    first = payload["each_gate"][0]
    assert set(["name", "priority", "passed", "reason", "metrics", "threshold"]).issubset(first.keys())
