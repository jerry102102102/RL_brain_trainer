from __future__ import annotations

import json

from hrl_trainer.v5_1.gates import DEFAULT_GATE, GateEvaluator, write_gate_report


def test_gate_evaluator_passes_when_thresholds_met() -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    result = evaluator.evaluate({"episodes": 4, "success_rate": 0.75, "intervention_rate": 0.25})

    assert result.passed is True
    assert result.reasons == []


def test_gate_evaluator_reports_fail_reasons() -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    result = evaluator.evaluate({"episodes": 2, "success_rate": 0.5, "intervention_rate": 0.5})

    assert result.passed is False
    assert "episodes<3" in result.reasons
    assert "success_rate<0.70" in result.reasons
    assert "intervention_rate>0.30" in result.reasons


def test_gate_report_is_written(tmp_path) -> None:
    evaluator = GateEvaluator(DEFAULT_GATE)
    result = evaluator.evaluate({"episodes": 3, "success_rate": 0.9, "intervention_rate": 0.0})

    path = write_gate_report(tmp_path / "gate.json", DEFAULT_GATE, result)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["gate"]["name"] == "v5_1_min_exec_gate"
    assert payload["result"]["passed"] is True
