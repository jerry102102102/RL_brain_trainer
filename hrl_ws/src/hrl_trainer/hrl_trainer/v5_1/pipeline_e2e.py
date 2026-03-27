"""V5.1 end-to-end pipeline (T6/T8).

Runs L1/L2/L3 smoke loop across episodes, applies curriculum and formal gates,
and emits inspectable artifacts/logs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .curriculum import CurriculumManager
from .gates import DEFAULT_GATE, GateEvaluator, write_gate_report
from .pipeline_smoke import run_smoke


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def run_pipeline_e2e(
    run_id: str,
    episodes: int,
    steps_per_episode: int,
    artifact_root: Path,
    enforce_gates: bool = False,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    logs_root = artifact_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    curriculum = CurriculumManager()
    gate_eval = GateEvaluator(DEFAULT_GATE)

    episodes_requested = max(1, int(episodes))
    successes = 0
    interventions = 0
    episode_outputs: list[dict[str, Any]] = []
    success_series: list[float] = []
    intervention_series: list[float] = []
    expected_log_lines_per_layer = 0
    reset_failures = 0

    for ep in range(episodes_requested):
        stage = curriculum.current_stage
        ep_id = f"{run_id}_ep{ep:03d}_{stage.name}"
        step_count = min(int(steps_per_episode), stage.step_budget)

        try:
            logs = run_smoke(run_id=ep_id, steps=step_count, log_root=logs_root, episode=ep)
        except Exception:
            reset_failures += 1
            break

        expected_log_lines_per_layer += max(1, int(step_count))

        # Deterministic pseudo-metrics for local smoke (keeps tests reproducible)
        success_rate = min(1.0, 0.50 + 0.10 * ep)
        record = curriculum.record_episode(success_rate=success_rate)
        ep_success = 1 if success_rate >= 0.70 else 0
        ep_intervention = 1 if success_rate < 0.60 else 0

        successes += ep_success
        interventions += ep_intervention
        success_series.append(success_rate)
        intervention_series.append(float(ep_intervention))

        episode_outputs.append(
            {
                "episode": ep,
                "run_id": ep_id,
                "stage": record.stage_name,
                "success_rate": success_rate,
                "promoted": record.promoted,
                "logs": logs,
                "has_intervention": bool(ep_intervention),
            }
        )

    metrics = {
        "episodes_requested": episodes_requested,
        "episodes_completed": len(episode_outputs),
        "success_rate": _safe_rate(successes, len(episode_outputs)),
        "intervention_rate": _safe_rate(interventions, len(episode_outputs)),
        "success_rate_first": success_series[0] if success_series else 0.0,
        "success_rate_last": success_series[-1] if success_series else 0.0,
        "intervention_rate_first": intervention_series[0] if intervention_series else 0.0,
        "intervention_rate_last": intervention_series[-1] if intervention_series else 0.0,
        "reset_failures": reset_failures,
        "log_lines_expected_per_layer": expected_log_lines_per_layer,
    }

    l1_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l1"])) for ep in episode_outputs)
    l2_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l2"])) for ep in episode_outputs)
    l3_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l3"])) for ep in episode_outputs)
    metrics.update(
        {
            "l1_log_lines": l1_lines,
            "l2_log_lines": l2_lines,
            "l3_log_lines": l3_lines,
        }
    )

    gate_result = gate_eval.evaluate(run_id=run_id, metrics=metrics)

    curriculum_path = artifact_root / "curriculum_state.json"
    gate_path = artifact_root / "gate_result.json"
    summary_path = artifact_root / "pipeline_summary.json"

    curriculum_path.write_text(
        json.dumps(curriculum.to_artifact(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_gate_report(gate_path, gate_result)

    summary = {
        "run_id": run_id,
        "timestamp_ns": time.time_ns(),
        "episodes": episode_outputs,
        "metrics": metrics,
        "artifacts": {
            "curriculum": str(curriculum_path),
            "gate": str(gate_path),
            "logs_root": str(logs_root),
        },
        "gate_overall_decision": gate_result.overall_decision,
        "gate_passed": gate_result.overall_decision == "GO",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status = "ok"
    exit_code = 0
    if enforce_gates and gate_result.overall_decision != "GO":
        status = "gates_blocked"
        exit_code = 2

    return {
        "summary": str(summary_path),
        "curriculum": str(curriculum_path),
        "gate": str(gate_path),
        "logs_root": str(logs_root),
        "status": status,
        "exit_code": exit_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V5.1 minimal e2e pipeline")
    parser.add_argument("--run-id", default=f"v5_1_e2e_{int(time.time())}")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int, default=5)
    parser.add_argument("--artifact-root", default="artifacts/v5_1/e2e")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()

    outputs = run_pipeline_e2e(
        run_id=args.run_id,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        artifact_root=Path(args.artifact_root),
        enforce_gates=args.enforce_gates,
    )
    print(json.dumps({"run_id": args.run_id, "outputs": outputs}, indent=2, sort_keys=True))
    return int(outputs.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
