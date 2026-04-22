"""Sequential Phase 1B pipeline: train approach, freeze, train dock, then switched eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .policy_config import repo_root, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sequential Phase 1B approach->dock->switched pipeline.")
    parser.add_argument("--run-id", default="phase1b_pipeline")
    parser.add_argument("--artifact-root")
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--dock-config", required=True)
    parser.add_argument("--switch-config")
    parser.add_argument("--approach-timesteps", type=int, default=65536)
    parser.add_argument("--dock-timesteps", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--approach-near-goal-threshold", type=float, default=0.80)
    parser.add_argument("--approach-success-threshold", type=float, default=0.0)
    return parser


def _run_module(module: str, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:  # pragma: no cover - CLI integration
    parser = build_arg_parser()
    args = parser.parse_args()
    root = repo_root()
    artifact_root = Path(args.artifact_root) if args.artifact_root else root / "artifacts" / "kinematic_phase1" / "phase1b_pipeline" / args.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)

    approach_root = artifact_root / "approach"
    dock_root = artifact_root / "dock"
    switched_root = artifact_root / "switched"

    _run_module(
        "hrl_trainer.kinematic_phase1.training.train_approach_policy",
        [
            "--config",
            args.approach_config,
            "--run-id",
            f"{args.run_id}_approach",
            "--artifact-root",
            str(approach_root),
            "--total-timesteps",
            str(args.approach_timesteps),
            "--seed",
            str(args.seed),
        ],
        root,
    )

    approach_summary_path = approach_root / "training_summary.json"
    approach_summary = json.loads(approach_summary_path.read_text())
    approach_eval = approach_summary.get("approach_eval_summary", {})
    near_goal_hit_rate = float(approach_eval.get("near_goal_hit_rate", 0.0))
    success_rate = float(approach_eval.get("success_rate", 0.0))
    gate_passed = (
        near_goal_hit_rate >= float(args.approach_near_goal_threshold)
        and success_rate >= float(args.approach_success_threshold)
    )

    pipeline_summary = {
        "run_id": args.run_id,
        "approach_artifact_root": str(approach_root),
        "dock_artifact_root": str(dock_root),
        "switched_artifact_root": str(switched_root),
        "approach_checkpoint": str(approach_root / "model_latest.zip"),
        "dock_checkpoint": str(dock_root / "model_latest.zip"),
        "approach_gate": {
            "passed": gate_passed,
            "near_goal_hit_rate": near_goal_hit_rate,
            "success_rate": success_rate,
            "required_near_goal_hit_rate": float(args.approach_near_goal_threshold),
            "required_success_rate": float(args.approach_success_threshold),
        },
    }

    if not gate_passed:
        write_json(artifact_root / "pipeline_summary.json", pipeline_summary)
        print(json.dumps(pipeline_summary, indent=2))
        return

    _run_module(
        "hrl_trainer.kinematic_phase1.training.train_dock_policy",
        [
            "--config",
            args.dock_config,
            "--run-id",
            f"{args.run_id}_dock",
            "--artifact-root",
            str(dock_root),
            "--total-timesteps",
            str(args.dock_timesteps),
            "--seed",
            str(args.seed),
        ],
        root,
    )

    switched_args = [
        "--approach-checkpoint",
        str(approach_root / "model_latest.zip"),
        "--dock-checkpoint",
        str(dock_root / "model_latest.zip"),
        "--approach-config",
        args.approach_config,
        "--artifact-root",
        str(switched_root),
        "--episodes",
        str(args.eval_episodes),
        "--seed",
        str(700001),
    ]
    if args.switch_config:
        switched_args.extend(["--switch-config", args.switch_config])

    _run_module(
        "hrl_trainer.kinematic_phase1.training.train_switched_eval",
        switched_args,
        root,
    )

    write_json(artifact_root / "pipeline_summary.json", pipeline_summary)
    print(json.dumps(pipeline_summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
