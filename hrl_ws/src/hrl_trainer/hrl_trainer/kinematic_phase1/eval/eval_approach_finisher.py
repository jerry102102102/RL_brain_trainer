"""Evaluate the simplified Approach -> Dock-Finisher pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..training.policy_config import approach_default_config_path, deep_merge, load_yaml_file, to_env_config, write_json
from .eval_deterministic import _load_sb3_model
from .eval_pipeline_ablation import _run_approach_with_handoff
from .eval_three_stage import _run_policy, _state_reset_options
from .fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable


def _mean(values: list[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _finisher_ready(result: dict[str, Any], *, cfg) -> bool:
    return bool(
        cfg.finisher_ready_pos_threshold_m > 0.0
        and cfg.finisher_ready_ori_threshold_rad > 0.0
        and float(result["final_position_error"]) <= cfg.finisher_ready_pos_threshold_m
        and float(result["final_orientation_error"]) <= cfg.finisher_ready_ori_threshold_rad
        and (cfg.finisher_ready_action_threshold <= 0.0 or float(result["final_action_magnitude"]) <= cfg.finisher_ready_action_threshold)
        and (cfg.finisher_ready_dq_threshold <= 0.0 or float(result["final_dq_norm"]) <= cfg.finisher_ready_dq_threshold)
    )


def _summarize_approach(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    first_ready_steps = [m["first_finisher_ready_step"] for m in metrics if m["first_finisher_ready_step"] is not None]
    return {
        "episode_count": len(metrics),
        "success_rate": _mean([m["success"] for m in metrics]),
        "finisher_ready_hit_rate": _mean([m["finisher_ready_hit"] for m in metrics]),
        "finisher_ready_dwell_rate": _mean([m["finisher_ready_dwell"] for m in metrics]),
        "same_step_finisher_ready_hit_rate": _mean([m["finisher_ready_hit"] for m in metrics]),
        "max_consecutive_finisher_ready_steps_mean": _mean([m["max_finisher_ready_streak"] for m in metrics]),
        "mean_time_to_finisher_ready": float(np.mean(first_ready_steps)) if first_ready_steps else None,
        "mean_final_position_error": _mean([m["final_position_error"] for m in metrics]),
        "mean_final_orientation_error": _mean([m["final_orientation_error"] for m in metrics]),
        "mean_min_position_error": _mean([m["min_position_error"] for m in metrics]),
        "mean_min_orientation_error": _mean([m["min_orientation_error"] for m in metrics]),
        "mean_final_action_magnitude": _mean([m["final_action_magnitude"] for m in metrics]),
        "mean_final_dq_norm": _mean([m["final_dq_norm"] for m in metrics]),
        "episode_metrics": metrics,
    }


def _summarize_finisher(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(metrics),
        "approach_to_finisher_success_rate": _mean([m["finisher_success"] for m in metrics]),
        "success_rate": _mean([m["finisher_success"] for m in metrics]),
        "handoff_count": int(sum(m["handoff_count"] for m in metrics)),
        "handoff_rate": _mean([m["handoff_count"] > 0 for m in metrics]),
        "handoff_ready_at_switch_rate": _mean([m["handoff_ready_at_switch"] for m in metrics]),
        "mean_first_handoff_step": _mean([m["first_handoff_step"] for m in metrics if m["first_handoff_step"] is not None]),
        "mean_handoff_position_error": _mean([m["handoff_position_error"] for m in metrics if m["handoff_position_error"] is not None]),
        "mean_handoff_orientation_error": _mean([m["handoff_orientation_error"] for m in metrics if m["handoff_orientation_error"] is not None]),
        "mean_handoff_action_magnitude": _mean([m["handoff_action_magnitude"] for m in metrics if m["handoff_action_magnitude"] is not None]),
        "mean_handoff_dq_norm": _mean([m["handoff_dq_norm"] for m in metrics if m["handoff_dq_norm"] is not None]),
        "mean_final_position_error": _mean([m["finisher_final_position_error"] for m in metrics]),
        "mean_final_orientation_error": _mean([m["finisher_final_orientation_error"] for m in metrics]),
        "mean_final_action_magnitude": _mean([m["finisher_final_action_magnitude"] for m in metrics]),
        "mean_final_dq_norm": _mean([m["finisher_final_dq_norm"] for m in metrics]),
        "episode_metrics": metrics,
    }


def evaluate_approach_finisher_pipeline(
    *,
    approach_checkpoint: Path,
    approach_algorithm: str,
    finisher_checkpoint: Path,
    finisher_algorithm: str,
    artifact_root: Path,
    approach_env_config: Phase1EnvConfig,
    finisher_env_config: Phase1EnvConfig,
    episodes: int,
    seed: int,
    stage_index: int,
    handoff_confirm_steps: int,
    handoff_mode: str,
) -> dict[str, Any]:
    approach_model = _load_sb3_model(approach_algorithm, approach_checkpoint)
    finisher_model = _load_sb3_model(finisher_algorithm, finisher_checkpoint)
    if approach_env_config.curriculum_config.enabled and approach_env_config.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(
            approach_env_config,
            seed=seed,
            stage_index=stage_index,
            n_episodes=episodes,
        )
        scope = "curriculum_region"
    else:
        suite = build_fixed_eval_suite(seed=seed, n_episodes=episodes, joint_specs=approach_env_config.joint_specs)
        scope = "fixed_random"

    approach_metrics: list[dict[str, Any]] = []
    finisher_metrics: list[dict[str, Any]] = []
    for episode in suite:
        approach_env = ArmKinematicEnv(config=approach_env_config)
        approach_env.set_curriculum_stage(stage_index)
        approach_env.set_policy_mode("approach")
        approach_result, handoff_result = _run_approach_with_handoff(
            env=approach_env,
            model=approach_model,
            reset_options={**episode.reset_options(), "policy_mode": "approach"},
            ready_cfg=approach_env_config.reward_config,
            handoff_confirm_steps=handoff_confirm_steps,
        )
        final_is_ready = _finisher_ready(approach_result, cfg=approach_env_config.reward_config)
        if handoff_mode == "final_settled":
            handoff_result = approach_result if final_is_ready else None
        elif handoff_mode == "final_always":
            handoff_result = approach_result

        approach_metrics.append(
            {
                "episode_id": episode.episode_id,
                "success": bool(approach_result["success"]),
                "finisher_ready_hit": bool(approach_result["dock_coarse_ready_hit"] or final_is_ready),
                "finisher_ready_dwell": bool(approach_result["dock_coarse_ready_dwell"] or final_is_ready),
                "max_finisher_ready_streak": int(approach_result["max_dock_coarse_ready_streak"]),
                "first_finisher_ready_step": approach_result["first_dock_coarse_ready_step"],
                "final_position_error": approach_result["final_position_error"],
                "final_orientation_error": approach_result["final_orientation_error"],
                "min_position_error": approach_result["min_position_error"],
                "min_orientation_error": approach_result["min_orientation_error"],
                "final_action_magnitude": approach_result["final_action_magnitude"],
                "final_dq_norm": approach_result["final_dq_norm"],
            }
        )

        finisher_result = None
        if handoff_result is not None:
            finisher_env = ArmKinematicEnv(config=finisher_env_config)
            finisher_env.set_policy_mode("dock")
            finisher_result = _run_policy(
                env=finisher_env,
                model=finisher_model,
                reset_options=_state_reset_options(handoff_result, policy_mode="dock"),
            )
        if finisher_result is None:
            finisher_result = {
                "success": False,
                "final_position_error": approach_result["final_position_error"],
                "final_orientation_error": approach_result["final_orientation_error"],
                "final_action_magnitude": approach_result["final_action_magnitude"],
                "final_dq_norm": approach_result["final_dq_norm"],
            }
        finisher_metrics.append(
            {
                "episode_id": episode.episode_id,
                "handoff_count": int(handoff_result is not None),
                "first_handoff_step": handoff_result["step_count"] if handoff_result else None,
                "handoff_ready_at_switch": bool(handoff_result is not None and _finisher_ready(handoff_result, cfg=approach_env_config.reward_config)),
                "handoff_position_error": handoff_result["final_position_error"] if handoff_result else None,
                "handoff_orientation_error": handoff_result["final_orientation_error"] if handoff_result else None,
                "handoff_action_magnitude": handoff_result["final_action_magnitude"] if handoff_result else None,
                "handoff_dq_norm": handoff_result["final_dq_norm"] if handoff_result else None,
                "finisher_success": bool(finisher_result["success"]),
                "finisher_final_position_error": finisher_result["final_position_error"],
                "finisher_final_orientation_error": finisher_result["final_orientation_error"],
                "finisher_final_action_magnitude": finisher_result["final_action_magnitude"],
                "finisher_final_dq_norm": finisher_result["final_dq_norm"],
            }
        )

    approach_summary = _summarize_approach(approach_metrics)
    finisher_summary = _summarize_finisher(finisher_metrics)
    combined = {
        "approach_checkpoint": str(approach_checkpoint),
        "finisher_checkpoint": str(finisher_checkpoint),
        "episodes": int(episodes),
        "seed": int(seed),
        "eval_scope": scope,
        "curriculum_stage_index": int(stage_index),
        "handoff_confirm_steps": int(handoff_confirm_steps),
        "handoff_mode": handoff_mode,
        "approach_only": {k: v for k, v in approach_summary.items() if k != "episode_metrics"},
        "approach_to_finisher": {k: v for k, v in finisher_summary.items() if k != "episode_metrics"},
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "approach_finisher_eval_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "approach_only_summary.json", approach_summary)
    write_json(artifact_root / "approach_to_finisher_summary.json", finisher_summary)
    write_json(artifact_root / "approach_finisher_pipeline_summary.json", combined)
    (artifact_root / "APPROACH_FINISHER_REPORT.md").write_text(_build_report(combined))
    return combined


def _build_report(summary: dict[str, Any]) -> str:
    approach = summary["approach_only"]
    pipeline = summary["approach_to_finisher"]
    conclusion = (
        "Approach -> Finisher is not yet reliable enough to be the final simplified pipeline."
        if pipeline["success_rate"] < 0.5
        else "Approach -> Finisher is becoming a viable simplified pipeline."
    )
    return "\n".join(
        [
            "# Approach -> Finisher Report",
            "",
            "Purpose: evaluate whether Approach can directly produce states that the frozen strict Dock-Finisher can consume.",
            f"Handoff mode: `{summary['handoff_mode']}`.",
            "",
            "## Approach Only",
            "",
            f"- Finisher-ready hit rate: {approach['finisher_ready_hit_rate']:.3f}",
            f"- Finisher-ready dwell rate: {approach['finisher_ready_dwell_rate']:.3f}",
            f"- Mean final position error: {approach['mean_final_position_error']:.6f}",
            f"- Mean final orientation error: {approach['mean_final_orientation_error']:.6f}",
            f"- Mean final action magnitude: {approach['mean_final_action_magnitude']:.6f}",
            f"- Mean final dq norm: {approach['mean_final_dq_norm']:.6f}",
            "",
            "## Approach -> Finisher",
            "",
            f"- Success rate: {pipeline['success_rate']:.3f}",
            f"- Handoff ready at switch rate: {pipeline['handoff_ready_at_switch_rate']:.3f}",
            f"- Mean handoff position error: {pipeline['mean_handoff_position_error']:.6f}",
            f"- Mean handoff orientation error: {pipeline['mean_handoff_orientation_error']:.6f}",
            f"- Mean handoff action magnitude: {pipeline['mean_handoff_action_magnitude']:.6f}",
            f"- Mean handoff dq norm: {pipeline['mean_handoff_dq_norm']:.6f}",
            "",
            "## Current Conclusion",
            "",
            conclusion,
            "",
        ]
    )


def _load_approach_config(config_path: str | None) -> dict[str, Any]:
    cfg = load_yaml_file(approach_default_config_path())
    if config_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(config_path)))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Approach -> frozen Dock-Finisher.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--finisher-checkpoint", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--approach-config")
    parser.add_argument("--finisher-config", required=True)
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--finisher-algorithm", default="td3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--handoff-confirm-steps", type=int, default=2)
    parser.add_argument("--handoff-mode", choices=("first_confirmed", "final_settled", "final_always"), default="final_settled")
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    summary = evaluate_approach_finisher_pipeline(
        approach_checkpoint=Path(args.approach_checkpoint),
        approach_algorithm=args.approach_algorithm,
        finisher_checkpoint=Path(args.finisher_checkpoint),
        finisher_algorithm=args.finisher_algorithm,
        artifact_root=Path(args.artifact_root),
        approach_env_config=to_env_config(_load_approach_config(args.approach_config)),
        finisher_env_config=to_env_config(load_yaml_file(Path(args.finisher_config))),
        episodes=args.episodes,
        seed=args.seed,
        stage_index=args.stage_index,
        handoff_confirm_steps=args.handoff_confirm_steps,
        handoff_mode=args.handoff_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
