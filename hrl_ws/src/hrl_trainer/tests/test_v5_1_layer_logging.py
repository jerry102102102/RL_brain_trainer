from __future__ import annotations

import json

from hrl_trainer.v5_1.log_summary import summarize_logs
from hrl_trainer.v5_1.pipeline_smoke import run_smoke


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_v5_1_layer_log_schema_fields_complete(tmp_path) -> None:
    outputs = run_smoke(run_id="t7_schema", steps=4, log_root=tmp_path / "logs", episode=2)

    l1 = _read_jsonl(outputs["l1"])
    l2 = _read_jsonl(outputs["l2"])
    l3 = _read_jsonl(outputs["l3"])

    assert len(l1) == len(l2) == len(l3) == 4

    l1_required = {"run_id", "episode", "step", "ts", "intent", "stage", "goal_summary"}
    l2_required = {"run_id", "episode", "step", "ts", "action_raw", "action_clipped", "delta_q", "policy_status"}
    l3_required = {"run_id", "episode", "step", "ts", "q_des", "q_actual", "intervention_type", "reason"}

    for rec in l1:
        payload = rec["payload"]
        assert l1_required.issubset(payload.keys())
        assert "gate_snapshot" in payload

    for rec in l2:
        payload = rec["payload"]
        assert l2_required.issubset(payload.keys())
        assert "gate_snapshot" in payload

    for rec in l3:
        payload = rec["payload"]
        assert l3_required.issubset(payload.keys())
        assert "gate_snapshot" in payload


def test_v5_1_log_summary_fields(tmp_path) -> None:
    run_smoke(run_id="t7_summary", steps=5, log_root=tmp_path / "logs", episode=1)

    summary = summarize_logs(tmp_path / "logs")

    assert "step_count" in summary
    assert "intervention_rate" in summary
    assert "action_saturation_rate" in summary
    assert "missing_fields" in summary

    assert summary["step_count"]["l1"] == 5
    assert summary["step_count"]["l2"] == 5
    assert summary["step_count"]["l3"] == 5

    assert isinstance(summary["intervention_rate"], float)
    assert isinstance(summary["action_saturation_rate"], float)

    assert set(summary["missing_fields"].keys()) == {"l1", "l2", "l3"}
