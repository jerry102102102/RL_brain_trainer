"""Deterministic eval gate for Workspace Expansion Curriculum runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..training.policy_config import (
    approach_default_config_path,
    config_dir,
    deep_merge,
    load_yaml_file,
    repo_root,
    to_env_config,
    write_json,
)
from ..workspace.workspace_curriculum import gate_config_from_dict, gated_score
from .eval_approach_finisher import _finisher_ready
from .eval_deterministic import _load_sb3_model
from .eval_pipeline_ablation import _run_approach_with_handoff
from .eval_three_stage import _run_policy, _state_reset_options
from .fixed_eval_suite import build_curriculum_local_eval_suite, suite_to_jsonable


def _mean(values: list[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_overlay_with_bases(path: Path) -> dict[str, Any]:
    overlay = load_yaml_file(path)
    base_config = overlay.pop("base_config", None)
    if not base_config:
        return overlay
    base_path = Path(str(base_config))
    if not base_path.is_absolute():
        candidate = path.parent / base_path
        cfg_candidate = config_dir() / base_path
        base_path = candidate if candidate.exists() else cfg_candidate if cfg_candidate.exists() else repo_root() / base_path
    return deep_merge(_load_overlay_with_bases(base_path), overlay)


def _failure_reason(*, result: dict[str, Any], ready_cfg, success: bool) -> str:
    if success:
        return "success"
    pos = float(result.get("final_position_error", 999.0))
    ori = float(result.get("final_orientation_error", 999.0))
    action = float(result.get("final_action_magnitude", 999.0))
    dq = float(result.get("final_dq_norm", 999.0))
    if pos > float(ready_cfg.finisher_ready_pos_threshold_m):
        return "position"
    if ori > float(ready_cfg.finisher_ready_ori_threshold_rad):
        return "orientation"
    if action > float(ready_cfg.finisher_ready_action_threshold):
        return "motion_action"
    if dq > float(ready_cfg.finisher_ready_dq_threshold):
        return "motion_dq"
    if not bool(result.get("dock_coarse_ready_dwell", False)):
        return "dwell"
    return "timeout_or_regression"


def _summarize_stage(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for item in metrics:
        reasons[item["failure_reason"]] = reasons.get(item["failure_reason"], 0) + 1
    return {
        "episode_count": len(metrics),
        "success_rate": _mean([m["success"] for m in metrics]),
        "finisher_ready_hit_rate": _mean([m["finisher_ready_hit"] for m in metrics]),
        "dwell_success_rate": _mean([m["finisher_ready_dwell"] for m in metrics]),
        "mean_final_position_error": _mean([m["final_position_error"] for m in metrics]),
        "mean_final_orientation_error": _mean([m["final_orientation_error"] for m in metrics]),
        "mean_final_action_magnitude": _mean([m["final_action_magnitude"] for m in metrics]),
        "mean_final_dq_norm": _mean([m["final_dq_norm"] for m in metrics]),
        "regression_rate": _mean([m["position_regression"] or m["orientation_regression"] for m in metrics]),
        "failure_reason_counts": reasons,
        "episode_metrics": metrics,
    }


def evaluate_workspace_expansion_checkpoint(
    *,
    approach_checkpoint: Path,
    approach_config_path: Path,
    artifact_root: Path,
    approach_algorithm: str = "ppo",
    finisher_checkpoint: Path | None = None,
    finisher_config_path: Path | None = None,
    finisher_algorithm: str = "ppo",
    episodes: int = 50,
    seed: int = 700001,
    stage_indices: list[int] | None = None,
    handoff_confirm_steps: int = 2,
    gate_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    approach_cfg = deep_merge(load_yaml_file(approach_default_config_path()), _load_overlay_with_bases(approach_config_path))
    approach_env_cfg = to_env_config(approach_cfg)
    finisher_env_cfg: Phase1EnvConfig | None = None
    if finisher_checkpoint and finisher_config_path:
        finisher_env_cfg = to_env_config(load_yaml_file(finisher_config_path))
    stage_count = len(approach_env_cfg.curriculum_config.stages)
    stages = stage_indices if stage_indices is not None else list(range(stage_count))
    stages = [int(np.clip(s, 0, stage_count - 1)) for s in stages]

    approach_model = _load_sb3_model(approach_algorithm, approach_checkpoint)
    finisher_model = _load_sb3_model(finisher_algorithm, finisher_checkpoint) if finisher_checkpoint else None

    artifact_root.mkdir(parents=True, exist_ok=True)
    stage_summaries: dict[int, dict[str, Any]] = {}
    target_rows: list[dict[str, Any]] = []
    worst_targets: list[dict[str, Any]] = []
    best_targets: list[dict[str, Any]] = []

    for stage_index in stages:
        suite = build_curriculum_local_eval_suite(
            approach_env_cfg,
            seed=seed + stage_index * 1009,
            stage_index=stage_index,
            n_episodes=episodes,
        )
        metrics: list[dict[str, Any]] = []
        for episode in suite:
            approach_env = ArmKinematicEnv(config=approach_env_cfg)
            approach_env.set_curriculum_stage(stage_index)
            approach_env.set_policy_mode("approach")
            approach_result, handoff_result = _run_approach_with_handoff(
                env=approach_env,
                model=approach_model,
                reset_options={**episode.reset_options(), "policy_mode": "approach"},
                ready_cfg=approach_env_cfg.reward_config,
                handoff_confirm_steps=handoff_confirm_steps,
            )
            final_ready = _finisher_ready(approach_result, cfg=approach_env_cfg.reward_config)
            handoff_result = approach_result if final_ready else handoff_result
            final_result = approach_result
            pipeline_success = bool(approach_result["success"])
            if finisher_model is not None and finisher_env_cfg is not None and handoff_result is not None:
                finisher_env = ArmKinematicEnv(config=finisher_env_cfg)
                finisher_env.set_policy_mode("dock")
                final_result = _run_policy(
                    env=finisher_env,
                    model=finisher_model,
                    reset_options=_state_reset_options(handoff_result, policy_mode="dock"),
                )
                pipeline_success = bool(final_result["success"])
            finisher_ready_hit = bool(approach_result["dock_coarse_ready_hit"] or final_ready)
            finisher_ready_dwell = bool(approach_result["dock_coarse_ready_dwell"] or final_ready)
            failure = _failure_reason(result=approach_result, ready_cfg=approach_env_cfg.reward_config, success=pipeline_success)
            pos_regression = float(approach_result["final_position_error"]) > float(approach_result["min_position_error"]) + 0.002
            ori_regression = float(approach_result["final_orientation_error"]) > float(approach_result["min_orientation_error"]) + 0.01
            row = {
                "episode_id": episode.episode_id,
                "stage_index": int(stage_index),
                "stage_name": approach_env_cfg.curriculum_config.stages[stage_index].name,
                "success": bool(pipeline_success),
                "finisher_ready_hit": finisher_ready_hit,
                "finisher_ready_dwell": finisher_ready_dwell,
                "failure_reason": failure,
                "final_position_error": float(final_result["final_position_error"]),
                "final_orientation_error": float(final_result["final_orientation_error"]),
                "approach_final_position_error": float(approach_result["final_position_error"]),
                "approach_final_orientation_error": float(approach_result["final_orientation_error"]),
                "final_action_magnitude": float(final_result["final_action_magnitude"]),
                "final_dq_norm": float(final_result["final_dq_norm"]),
                "min_position_error": float(approach_result["min_position_error"]),
                "min_orientation_error": float(approach_result["min_orientation_error"]),
                "position_regression": bool(pos_regression),
                "orientation_regression": bool(ori_regression),
                "goal_position": list(np.asarray(episode.goal_pose6[:3], dtype=float)),
                "goal_orientation": list(np.asarray(episode.goal_pose6[3:], dtype=float)),
            }
            metrics.append(row)
            target_rows.append(row)
        summary = _summarize_stage(metrics)
        stage_summaries[stage_index] = {k: v for k, v in summary.items() if k != "episode_metrics"}
        worst_targets.extend(sorted(metrics, key=lambda x: x["final_position_error"] + 0.02 * x["final_orientation_error"], reverse=True)[:5])
        best_targets.extend(sorted(metrics, key=lambda x: x["final_position_error"] + 0.02 * x["final_orientation_error"])[:5])
        write_json(artifact_root / f"stage_{stage_index:02d}_suite.json", {"suite": suite_to_jsonable(suite)})
        write_json(artifact_root / f"stage_{stage_index:02d}_metrics.json", summary)

    cfg = gate_config_from_dict(gate_config)
    score_stage_index = int((gate_config or {}).get("score_stage_index", max(stages)))
    score_stage_index = int(np.clip(score_stage_index, min(stages), max(stages)))
    selection = gated_score(stage_summaries, score_stage_index, cfg)
    failure_report = {
        "stage_failure_reason_counts": {
            str(idx): data.get("failure_reason_counts", {}) for idx, data in stage_summaries.items()
        },
        "worst_targets": worst_targets[:50],
        "best_targets": best_targets[:50],
    }
    payload = {
        "approach_checkpoint": str(approach_checkpoint),
        "finisher_checkpoint": str(finisher_checkpoint) if finisher_checkpoint else None,
        "episodes_per_stage": int(episodes),
        "seed": int(seed),
        "stage_metrics": {str(k): v for k, v in stage_summaries.items()},
        "best_model_selection": selection,
        "workspace_failure_report": failure_report,
        "target_rows": target_rows,
    }
    write_json(artifact_root / "stage_metrics.json", payload["stage_metrics"])
    write_json(artifact_root / "workspace_failure_report.json", failure_report)
    write_json(artifact_root / "best_model_selection_summary.json", selection)
    write_json(artifact_root / "workspace_eval_summary.json", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate workspace expansion curriculum checkpoint.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--finisher-checkpoint")
    parser.add_argument("--finisher-config")
    parser.add_argument("--finisher-algorithm", default="ppo")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--stages", default="")
    parser.add_argument("--handoff-confirm-steps", type=int, default=2)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    stages = [int(x) for x in args.stages.split(",") if x.strip()] if args.stages else None
    summary = evaluate_workspace_expansion_checkpoint(
        approach_checkpoint=Path(args.approach_checkpoint),
        approach_config_path=Path(args.approach_config),
        artifact_root=Path(args.artifact_root),
        approach_algorithm=args.approach_algorithm,
        finisher_checkpoint=Path(args.finisher_checkpoint) if args.finisher_checkpoint else None,
        finisher_config_path=Path(args.finisher_config) if args.finisher_config else None,
        finisher_algorithm=args.finisher_algorithm,
        episodes=args.episodes,
        seed=args.seed,
        stage_indices=stages,
        handoff_confirm_steps=args.handoff_confirm_steps,
    )
    printable = {k: v for k, v in summary.items() if k != "target_rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
