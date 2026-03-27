from __future__ import annotations

import json

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
    assert summary_path.exists()
    assert curriculum_path.exists()
    assert gate_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary["episodes"]) == 4
    assert "metrics" in summary
    assert "artifacts" in summary

    logs_root = tmp_path / "artifacts" / "logs"
    assert (logs_root / "l1").exists()
    assert (logs_root / "l2").exists()
    assert (logs_root / "l3").exists()

    l1_any = list((logs_root / "l1").glob("*.jsonl"))
    l2_any = list((logs_root / "l2").glob("*.jsonl"))
    l3_any = list((logs_root / "l3").glob("*.jsonl"))
    assert l1_any and l2_any and l3_any
