"""Alternating joint fine-tuning for separate approach and dock policies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .policy_config import repo_root, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alternating switched joint fine-tune for PPO approach and TD3 dock policies."
    )
    parser.add_argument("--run-id", default="joint_switched_finetune")
    parser.add_argument("--artifact-root")
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--dock-config", required=True)
    parser.add_argument("--switch-config", required=True)
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--dock-checkpoint", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--approach-cycle-timesteps", type=int, default=131072)
    parser.add_argument("--dock-cycle-timesteps", type=int, default=262144)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _run_module(module: str, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:  # pragma: no cover - CLI integration
    parser = build_arg_parser()
    args = parser.parse_args()

    root = repo_root()
    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else root / "artifacts" / "kinematic_phase1" / "phase1b" / args.run_id
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    current_approach_checkpoint = str(Path(args.approach_checkpoint))
    current_dock_checkpoint = str(Path(args.dock_checkpoint))
    cycles_summary: list[dict[str, Any]] = []

    for cycle_idx in range(1, int(args.cycles) + 1):
        cycle_root = artifact_root / f"cycle_{cycle_idx:02d}"
        approach_root = cycle_root / "approach"
        dock_root = cycle_root / "dock"
        switched_root = cycle_root / "switched"
        cycle_root.mkdir(parents=True, exist_ok=True)

        _run_module(
            "hrl_trainer.kinematic_phase1.training.train_approach_policy",
            [
                "--config",
                args.approach_config,
                "--run-id",
                f"{args.run_id}_approach_c{cycle_idx:02d}",
                "--artifact-root",
                str(approach_root),
                "--total-timesteps",
                str(args.approach_cycle_timesteps),
                "--seed",
                str(args.seed),
                "--resume-from",
                current_approach_checkpoint,
            ],
            root,
        )
        current_approach_checkpoint = str(approach_root / "model_latest.zip")
        approach_summary = _read_json(approach_root / "training_summary.json")

        _run_module(
            "hrl_trainer.kinematic_phase1.training.train_dock_td3_policy",
            [
                "--config",
                args.dock_config,
                "--run-id",
                f"{args.run_id}_dock_c{cycle_idx:02d}",
                "--artifact-root",
                str(dock_root),
                "--total-timesteps",
                str(args.dock_cycle_timesteps),
                "--seed",
                str(args.seed),
                "--resume-from",
                current_dock_checkpoint,
            ],
            root,
        )
        current_dock_checkpoint = str(dock_root / "model_latest.zip")
        dock_summary = _read_json(dock_root / "training_summary.json")

        _run_module(
            "hrl_trainer.kinematic_phase1.training.train_switched_eval",
            [
                "--approach-checkpoint",
                current_approach_checkpoint,
                "--dock-checkpoint",
                current_dock_checkpoint,
                "--approach-config",
                args.approach_config,
                "--dock-config",
                args.dock_config,
                "--switch-config",
                args.switch_config,
                "--approach-algorithm",
                "ppo",
                "--dock-algorithm",
                "td3",
                "--artifact-root",
                str(switched_root),
                "--seed",
                str(700001 + cycle_idx),
                "--episodes",
                str(args.eval_episodes),
            ],
            root,
        )
        switched_summary = _read_json(switched_root / "switched_eval_summary.json")

        cycle_summary = {
            "cycle_index": cycle_idx,
            "approach_checkpoint": current_approach_checkpoint,
            "dock_checkpoint": current_dock_checkpoint,
            "approach_eval_summary": approach_summary.get("approach_eval_summary", {}),
            "dock_eval_summary": dock_summary.get("dock_eval_summary", {}),
            "dock_reverse_curriculum_summary": dock_summary.get("dock_reverse_curriculum_summary"),
            "switched_eval_summary": switched_summary,
        }
        cycles_summary.append(cycle_summary)
        write_json(cycle_root / "cycle_summary.json", cycle_summary)

    final_summary = {
        "run_id": args.run_id,
        "approach_config": args.approach_config,
        "dock_config": args.dock_config,
        "switch_config": args.switch_config,
        "initial_approach_checkpoint": str(Path(args.approach_checkpoint)),
        "initial_dock_checkpoint": str(Path(args.dock_checkpoint)),
        "final_approach_checkpoint": current_approach_checkpoint,
        "final_dock_checkpoint": current_dock_checkpoint,
        "cycles": cycles_summary,
    }
    write_json(artifact_root / "joint_training_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
