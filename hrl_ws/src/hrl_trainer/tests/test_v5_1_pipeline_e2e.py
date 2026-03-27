from __future__ import annotations

import json

from hrl_trainer.v5_1 import pipeline_e2e
from hrl_trainer.v5_1.pipeline_e2e import run_pipeline_e2e


def test_pipeline_e2e_outputs_artifacts_and_logs(tmp_path) -> None:
    out = run_pipeline_e2e(
        run_id="test_e2e",
        episodes=4,
        steps_per_episode=3,
        artifact_root=tmp_path / "artifacts",
    )

    summary_path = tmp_path / "artifacts" / "pipeline_summary.json"
    curriculum_path = tmp_path / "artifacts" / "curriculum_state.json"
    gate_path = tmp_path / "artifacts" / "gate_result.json"

    assert out["summary"] == str(summary_path)
    assert out["status"] == "ok"
    assert out["exit_code"] == 0
    assert summary_path.exists()
    assert curriculum_path.exists()
    assert gate_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary["episodes"]) == 4
    assert "metrics" in summary
    assert "artifacts" in summary
    assert summary["gate_overall_decision"] == "GO"

    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate_payload["overall_decision"] == "GO"

    logs_root = tmp_path / "artifacts" / "logs"
    assert (logs_root / "l1").exists()
    assert (logs_root / "l2").exists()
    assert (logs_root / "l3").exists()


def test_pipeline_e2e_enforce_gates_returns_nonzero_on_fail(tmp_path, monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(pipeline_e2e, "run_smoke", _boom)

    out = run_pipeline_e2e(
        run_id="test_e2e_fail",
        episodes=2,
        steps_per_episode=3,
        artifact_root=tmp_path / "artifacts",
        enforce_gates=True,
    )

    assert out["status"] == "gates_blocked"
    assert out["exit_code"] != 0

    gate_payload = json.loads((tmp_path / "artifacts" / "gate_result.json").read_text(encoding="utf-8"))
    assert gate_payload["overall_decision"] == "HOLD"
