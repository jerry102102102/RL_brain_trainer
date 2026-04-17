"""Deterministic artifact writers for v5 trainer loop outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .trainer_loop import V5EpisodeSummary, run_v5_training_episode

ARTIFACT_SCHEMA = "v5_trainer_loop_artifact"
ARTIFACT_VERSION = "1.1"


def _sorted_float_mapping(values: dict[str, float]) -> dict[str, float]:
    return {key: float(values[key]) for key in sorted(values)}


def _build_step_rows(summary: V5EpisodeSummary) -> list[dict[str, Any]]:
    return [
        {
            "step_index": int(step.step_index),
            "terminal": bool(step.terminal),
            "total_reward": float(step.total_reward),
            "train_counter": int(step.train_counter),
            "update_counter": int(step.update_counter),
            "weighted_terms": _sorted_float_mapping(step.weighted_terms),
        }
        for step in summary.steps
    ]


def build_v5_episode_artifact(
    summary: V5EpisodeSummary,
    *,
    rollout_skill_sequence: Sequence[str] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build deterministic artifact payload from a v5 episode summary."""
    return {
        "metadata": {
            "schema": ARTIFACT_SCHEMA,
            "version": ARTIFACT_VERSION,
            "timestamp": timestamp,
        },
        "episode_index": int(summary.episode_index),
        "stage_id": summary.stage_id,
        "selected_policy": summary.selected_policy,
        "reward_term_totals": _sorted_float_mapping(summary.reward_term_totals),
        "total_reward": float(summary.total_reward),
        "train_counters": {
            "train_episode_calls": int(summary.train_counters.train_episode_calls),
            "train_step_calls": int(summary.train_counters.train_step_calls),
            "update_episode_calls": int(summary.train_counters.update_episode_calls),
            "update_step_calls": int(summary.train_counters.update_step_calls),
        },
        "per_step_weighted_terms": _build_step_rows(summary),
        "rollout_skill_sequence": list(rollout_skill_sequence or []),
    }


def write_v5_episode_artifacts(
    summary: V5EpisodeSummary,
    *,
    json_path: str | Path,
    jsonl_path: str | Path,
    rollout_skill_sequence: Sequence[str] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Write deterministic JSON + JSONL artifacts for one episode."""
    artifact = build_v5_episode_artifact(
        summary,
        rollout_skill_sequence=rollout_skill_sequence,
        timestamp=timestamp,
    )

    json_target = Path(json_path)
    jsonl_target = Path(jsonl_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    jsonl_target.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    json_target.write_text(json_text, encoding="utf-8")

    jsonl_rows: list[dict[str, Any]] = [
        {
            "episode_index": artifact["episode_index"],
            "record_type": "episode_summary",
            "reward_term_totals": artifact["reward_term_totals"],
            "rollout_skill_sequence": artifact["rollout_skill_sequence"],
            "selected_policy": artifact["selected_policy"],
            "stage_id": artifact["stage_id"],
            "total_reward": artifact["total_reward"],
            "train_counters": artifact["train_counters"],
        }
    ]
    jsonl_rows.extend(
        {
            "episode_index": artifact["episode_index"],
            "record_type": "step_weighted_terms",
            "stage_id": artifact["stage_id"],
            "selected_policy": artifact["selected_policy"],
            **row,
        }
        for row in artifact["per_step_weighted_terms"]
    )
    jsonl_text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in jsonl_rows) + "\n"
    jsonl_target.write_text(jsonl_text, encoding="utf-8")

    return artifact


def _build_smoke_step_inputs(step_count: int, terminal_success: bool) -> list[dict[str, Any]]:
    effective_steps = max(1, int(step_count))
    rows: list[dict[str, Any]] = []
    for index in range(effective_steps):
        is_terminal = index == (effective_steps - 1)
        rows.append(
            {
                "potential_current": float(index),
                "potential_next": float(index + 1),
                "coverage": 0.05,
                "subgoal": 0.1,
                "safety_violation": 0.0,
                "action_delta": 0.1,
                "terminal": is_terminal,
                "terminal_success": terminal_success,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write deterministic v5 trainer-loop artifacts")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1, help="Run multiple synthetic episodes for smoke coverage.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--terminal-success", action="store_true")
    parser.add_argument("--policy-id", default="rule_l2_v0", choices=["rule_l2", "rule_l2_v0", "rl_l2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-path", default="/tmp/v5_m2_4_smoke_artifact.json")
    parser.add_argument("--jsonl-path", default="/tmp/v5_m2_4_smoke_artifact.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="When set and episodes>1, writes per-episode artifacts into this directory.",
    )
    parser.add_argument("--timestamp", default=None, help="Optional metadata timestamp for explicit reproducibility")
    parser.add_argument(
        "--rollout-skill-sequence",
        default="APPROACH,INSERT_SUPPORT,LIFT_CARRY,PLACE,RETREAT",
        help="Comma-separated skill sequence to include in artifact",
    )
    args = parser.parse_args(argv)

    sequence = [token.strip() for token in args.rollout_skill_sequence.split(",") if token.strip()]
    if args.output_dir and args.episodes > 1:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, Any]] = []
        for offset in range(max(1, int(args.episodes))):
            episode_index = int(args.episode_index) + offset
            summary = run_v5_training_episode(
                episode_index=episode_index,
                step_inputs=_build_smoke_step_inputs(args.steps, args.terminal_success),
                policy_id=args.policy_id,
                seed=args.seed,
            )
            json_target = output_dir / f"episode_{episode_index:04d}.json"
            jsonl_target = output_dir / f"episode_{episode_index:04d}.jsonl"
            artifacts.append(
                write_v5_episode_artifacts(
                    summary,
                    json_path=json_target,
                    jsonl_path=jsonl_target,
                    rollout_skill_sequence=sequence,
                    timestamp=args.timestamp,
                )
            )
        print(json.dumps({"output_dir": str(output_dir), "episodes": len(artifacts)}, indent=2, sort_keys=True))
        return 0

    summary = run_v5_training_episode(
        episode_index=args.episode_index,
        step_inputs=_build_smoke_step_inputs(args.steps, args.terminal_success),
        policy_id=args.policy_id,
        seed=args.seed,
    )

    artifact = write_v5_episode_artifacts(
        summary,
        json_path=args.json_path,
        jsonl_path=args.jsonl_path,
        rollout_skill_sequence=sequence,
        timestamp=args.timestamp,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
