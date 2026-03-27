"""V5.1 end-to-end pipeline (T6).

Runs L1/L2/L3 smoke loop across episodes, applies curriculum and acceptance gates,
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


def run_pipeline_e2e(
    run_id: str,
    episodes: int,
    steps_per_episode: int,
    artifact_root: Path,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    logs_root = artifact_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    curriculum = CurriculumManager()
    gate_eval = GateEvaluator(DEFAULT_GATE)

    successes = 0
    interventions = 0
    episode_outputs: list[dict[str, Any]] = []

    for ep in range(max(1, int(episodes))):
        stage = curriculum.current_stage
        ep_id = f"{run_id}_ep{ep:03d}_{stage.name}"
        logs = run_smoke(run_id=ep_id, steps=min(int(steps_per_episode), stage.step_budget), log_root=logs_root)

        # Deterministic pseudo-metrics for local smoke (keeps tests reproducible)
        success_rate = min(1.0, 0.50 + 0.10 * ep)
        record = curriculum.record_episode(success_rate=success_rate)
        ep_success = 1 if success_rate >= 0.70 else 0
        ep_intervention = 1 if success_rate < 0.60 else 0

        successes += ep_success
        interventions += ep_intervention

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
        "episodes": len(episode_outputs),
        "success_rate": _safe_rate(successes, len(episode_outputs)),
        "intervention_rate": _safe_rate(interventions, len(episode_outputs)),
    }
    gate_result = gate_eval.evaluate(metrics)

    curriculum_path = artifact_root / "curriculum_state.json"
    gate_path = artifact_root / "gate_result.json"
    summary_path = artifact_root / "pipeline_summary.json"

    curriculum_path.write_text(
        json.dumps(curriculum.to_artifact(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_gate_report(gate_path, DEFAULT_GATE, gate_result)

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
        "gate_passed": gate_result.passed,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "summary": str(summary_path),
        "curriculum": str(curriculum_path),
        "gate": str(gate_path),
        "logs_root": str(logs_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V5.1 minimal e2e pipeline")
    parser.add_argument("--run-id", default=f"v5_1_e2e_{int(time.time())}")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int, default=5)
    parser.add_argument("--artifact-root", default="artifacts/v5_1/e2e")
    args = parser.parse_args()

    outputs = run_pipeline_e2e(
        run_id=args.run_id,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        artifact_root=Path(args.artifact_root),
    )
    print(json.dumps({"run_id": args.run_id, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
