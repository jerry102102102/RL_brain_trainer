"""Pipeline ablations for Approach, Dock-Coarse, and Dock-Finisher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..training.policy_config import (
    approach_default_config_path,
    deep_merge,
    dock_coarse_default_config_path,
    load_yaml_file,
    to_env_config,
    write_json,
)
from .eval_deterministic import _load_sb3_model
from .eval_three_stage import _dock_coarse_ready, _predict, _run_policy, _state_reset_options
from .fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable


def _mean(values: list[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _ready_state(
    result: dict[str, Any],
    *,
    cfg,
) -> bool:
    return _dock_coarse_ready(
        pos=float(result["final_position_error"]),
        ori=float(result["final_orientation_error"]),
        action_norm=float(result["final_action_magnitude"]),
        dq_norm=float(result["final_dq_norm"]),
        cfg=cfg,
    )


def _strict_like_state(
    result: dict[str, Any],
    *,
    pos_threshold: float,
    ori_threshold: float,
    action_threshold: float,
    dq_threshold: float,
) -> bool:
    return bool(
        float(result["final_position_error"]) <= pos_threshold
        and float(result["final_orientation_error"]) <= ori_threshold
        and (action_threshold <= 0.0 or float(result["final_action_magnitude"]) <= action_threshold)
        and (dq_threshold <= 0.0 or float(result["final_dq_norm"]) <= dq_threshold)
    )


def _run_approach_with_handoff(
    *,
    env: ArmKinematicEnv,
    model: Any,
    reset_options: dict[str, Any],
    ready_cfg,
    handoff_confirm_steps: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    obs, info = env.reset(options=reset_options)
    terminated = False
    truncated = False
    steps = 0
    min_pos = float(info["position_error_norm"])
    min_ori = float(info["orientation_error_norm"])
    action_norms: list[float] = []
    dq_norms: list[float] = []
    ready_hit = False
    ready_streak = 0
    max_ready_streak = 0
    first_ready_step: int | None = None
    handoff_result: dict[str, Any] | None = None

    while not (terminated or truncated):
        action = _predict(model, obs)
        action_norm = float(np.linalg.norm(action))
        action_norms.append(action_norm)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        dq_norm = float(info.get("executed_delta_q_l2", np.linalg.norm(info["dq"])))
        dq_norms.append(dq_norm)
        pos = float(info["position_error_norm"])
        ori = float(info["orientation_error_norm"])
        min_pos = min(min_pos, pos)
        min_ori = min(min_ori, ori)
        if _dock_coarse_ready(pos=pos, ori=ori, action_norm=action_norm, dq_norm=dq_norm, cfg=ready_cfg):
            ready_hit = True
            if first_ready_step is None:
                first_ready_step = steps
            ready_streak += 1
        else:
            ready_streak = 0
        max_ready_streak = max(max_ready_streak, ready_streak)

        if handoff_result is None and ready_streak >= handoff_confirm_steps:
            handoff_result = {
                "success": bool(info["success"]),
                "final_position_error": pos,
                "final_orientation_error": ori,
                "min_position_error": min_pos,
                "min_orientation_error": min_ori,
                "final_action_magnitude": action_norm,
                "final_dq_norm": dq_norm,
                "mean_action_magnitude": float(np.mean(action_norms)) if action_norms else 0.0,
                "mean_dq_norm": float(np.mean(dq_norms)) if dq_norms else 0.0,
                "dock_coarse_ready_hit": bool(ready_hit),
                "dock_coarse_ready_dwell": bool(max_ready_streak >= handoff_confirm_steps),
                "max_dock_coarse_ready_streak": int(max_ready_streak),
                "first_dock_coarse_ready_step": first_ready_step,
                "step_count": int(steps),
                "final_q": np.asarray(info["q"], dtype=float).tolist(),
                "final_dq": np.asarray(info["dq"], dtype=float).tolist(),
                "final_prev_action": np.asarray(env._prev_action, dtype=float).tolist(),  # noqa: SLF001 - eval handoff state
                "goal_q": np.asarray(info["goal_q"], dtype=float).tolist(),
                "goal_pose6": np.asarray(info["goal_pose6"], dtype=float).tolist(),
            }

    approach_result = {
        "success": bool(info["success"]),
        "final_position_error": float(info["position_error_norm"]),
        "final_orientation_error": float(info["orientation_error_norm"]),
        "min_position_error": min_pos,
        "min_orientation_error": min_ori,
        "final_action_magnitude": float(action_norms[-1]) if action_norms else 0.0,
        "final_dq_norm": float(dq_norms[-1]) if dq_norms else 0.0,
        "mean_action_magnitude": float(np.mean(action_norms)) if action_norms else 0.0,
        "mean_dq_norm": float(np.mean(dq_norms)) if dq_norms else 0.0,
        "dock_coarse_ready_hit": bool(ready_hit),
        "dock_coarse_ready_dwell": bool(max_ready_streak >= handoff_confirm_steps),
        "max_dock_coarse_ready_streak": int(max_ready_streak),
        "first_dock_coarse_ready_step": first_ready_step,
        "step_count": int(steps),
        "final_q": np.asarray(info["q"], dtype=float).tolist(),
        "final_dq": np.asarray(info["dq"], dtype=float).tolist(),
        "final_prev_action": np.asarray(env._prev_action, dtype=float).tolist(),  # noqa: SLF001 - eval state
        "goal_q": np.asarray(info["goal_q"], dtype=float).tolist(),
        "goal_pose6": np.asarray(info["goal_pose6"], dtype=float).tolist(),
    }
    return approach_result, handoff_result


def _pipeline_summary(metrics: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    successes = [bool(m.get(f"{prefix}_success", False)) for m in metrics]
    regressions = [
        bool(m[f"{prefix}_final_position_error"] > m[f"{prefix}_min_position_error"] + 0.001)
        for m in metrics
        if f"{prefix}_min_position_error" in m
    ]
    return {
        "overall_success_rate": _mean(successes),
        "final_position_error_mean": _mean([m[f"{prefix}_final_position_error"] for m in metrics]),
        "final_orientation_error_mean": _mean([m[f"{prefix}_final_orientation_error"] for m in metrics]),
        "dwell_success_rate": _mean([m.get(f"{prefix}_dwell_success", m.get(f"{prefix}_success", False)) for m in metrics]),
        "regression_rate": _mean(regressions),
        "final_action_magnitude_mean": _mean([m[f"{prefix}_final_action_magnitude"] for m in metrics]),
        "final_dq_norm_mean": _mean([m[f"{prefix}_final_dq_norm"] for m in metrics]),
    }


def evaluate_pipeline_ablation(
    *,
    approach_checkpoint: Path,
    approach_algorithm: str,
    dock_coarse_checkpoint: Path,
    dock_coarse_algorithm: str,
    finisher_checkpoint: Path,
    finisher_algorithm: str,
    artifact_root: Path,
    approach_env_config: Phase1EnvConfig,
    dock_coarse_env_config: Phase1EnvConfig,
    finisher_env_config: Phase1EnvConfig,
    episodes: int,
    seed: int,
    stage_index: int,
    handoff_confirm_steps: int,
    handoff_mode: str,
    finisher_direct_pos_threshold: float,
    finisher_direct_ori_threshold: float,
    finisher_direct_action_threshold: float,
    finisher_direct_dq_threshold: float,
    strict_like_pos_threshold: float,
    strict_like_ori_threshold: float,
    strict_like_action_threshold: float,
    strict_like_dq_threshold: float,
) -> dict[str, Any]:
    approach_model = _load_sb3_model(approach_algorithm, approach_checkpoint)
    dock_coarse_model = _load_sb3_model(dock_coarse_algorithm, dock_coarse_checkpoint)
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

    episode_metrics: list[dict[str, Any]] = []
    delta_metrics: list[dict[str, Any]] = []
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
        if handoff_mode == "final_settled":
            handoff_result = approach_result if _ready_state(approach_result, cfg=approach_env_config.reward_config) else None

        handoff_count = 1 if handoff_result is not None else 0
        direct_result = None
        coarse_result = None
        coarse_then_finisher_result = None
        conditional_result = None
        conditional_route = "none"
        if handoff_result is not None:
            direct_env = ArmKinematicEnv(config=finisher_env_config)
            direct_env.set_policy_mode("dock")
            direct_result = _run_policy(
                env=direct_env,
                model=finisher_model,
                reset_options=_state_reset_options(handoff_result, policy_mode="dock"),
            )

            coarse_env = ArmKinematicEnv(config=dock_coarse_env_config)
            coarse_env.set_policy_mode("dock_coarse")
            coarse_result = _run_policy(
                env=coarse_env,
                model=dock_coarse_model,
                reset_options=_state_reset_options(handoff_result, policy_mode="dock_coarse"),
            )
            finisher_env = ArmKinematicEnv(config=finisher_env_config)
            finisher_env.set_policy_mode("dock")
            coarse_then_finisher_result = _run_policy(
                env=finisher_env,
                model=finisher_model,
                reset_options=_state_reset_options(coarse_result, policy_mode="dock"),
            )

            use_direct = _strict_like_state(
                handoff_result,
                pos_threshold=finisher_direct_pos_threshold,
                ori_threshold=finisher_direct_ori_threshold,
                action_threshold=finisher_direct_action_threshold,
                dq_threshold=finisher_direct_dq_threshold,
            )
            conditional_route = "direct_finisher" if use_direct else "dock_coarse_then_finisher"
            conditional_result = direct_result if use_direct else coarse_then_finisher_result

            pre_ready = _ready_state(handoff_result, cfg=approach_env_config.reward_config)
            post_ready = _ready_state(coarse_result, cfg=approach_env_config.reward_config)
            pre_strict_like = _strict_like_state(
                handoff_result,
                pos_threshold=strict_like_pos_threshold,
                ori_threshold=strict_like_ori_threshold,
                action_threshold=strict_like_action_threshold,
                dq_threshold=strict_like_dq_threshold,
            )
            post_strict_like = _strict_like_state(
                coarse_result,
                pos_threshold=strict_like_pos_threshold,
                ori_threshold=strict_like_ori_threshold,
                action_threshold=strict_like_action_threshold,
                dq_threshold=strict_like_dq_threshold,
            )
            delta_metrics.append(
                {
                    "episode_id": episode.episode_id,
                    "delta_position_error_after_coarse": coarse_result["final_position_error"] - handoff_result["final_position_error"],
                    "delta_orientation_error_after_coarse": coarse_result["final_orientation_error"] - handoff_result["final_orientation_error"],
                    "delta_action_magnitude_after_coarse": coarse_result["final_action_magnitude"] - handoff_result["final_action_magnitude"],
                    "delta_dq_norm_after_coarse": coarse_result["final_dq_norm"] - handoff_result["final_dq_norm"],
                    "pre_handoff_ready": pre_ready,
                    "post_handoff_ready": post_ready,
                    "handoff_ready_preserved": pre_ready and post_ready,
                    "handoff_ready_destroyed": pre_ready and not post_ready,
                    "pre_strict_like": pre_strict_like,
                    "post_strict_like": post_strict_like,
                    "strict_like_preserved": pre_strict_like and post_strict_like,
                    "strict_like_destroyed": pre_strict_like and not post_strict_like,
                }
            )

        def _empty_result() -> dict[str, Any]:
            return {
                "success": False,
                "final_position_error": approach_result["final_position_error"],
                "final_orientation_error": approach_result["final_orientation_error"],
                "min_position_error": approach_result["min_position_error"],
                "final_action_magnitude": approach_result["final_action_magnitude"],
                "final_dq_norm": approach_result["final_dq_norm"],
            }

        direct = direct_result or _empty_result()
        coarse_final = coarse_then_finisher_result or _empty_result()
        conditional = conditional_result or _empty_result()
        episode_metrics.append(
            {
                "episode_id": episode.episode_id,
                "handoff_count": handoff_count,
                "first_handoff_step": handoff_result["step_count"] if handoff_result else None,
                "handoff_ready_at_switch": bool(handoff_result is not None and _ready_state(handoff_result, cfg=approach_env_config.reward_config)),
                "conditional_route": conditional_route,
                "approach_only_success": bool(approach_result["success"]),
                "approach_only_dwell_success": bool(approach_result["dock_coarse_ready_dwell"]),
                "approach_only_final_position_error": approach_result["final_position_error"],
                "approach_only_final_orientation_error": approach_result["final_orientation_error"],
                "approach_only_min_position_error": approach_result["min_position_error"],
                "approach_only_final_action_magnitude": approach_result["final_action_magnitude"],
                "approach_only_final_dq_norm": approach_result["final_dq_norm"],
                "approach_to_finisher_success": bool(direct["success"]),
                "approach_to_finisher_dwell_success": bool(direct["success"]),
                "approach_to_finisher_final_position_error": direct["final_position_error"],
                "approach_to_finisher_final_orientation_error": direct["final_orientation_error"],
                "approach_to_finisher_min_position_error": direct["min_position_error"],
                "approach_to_finisher_final_action_magnitude": direct["final_action_magnitude"],
                "approach_to_finisher_final_dq_norm": direct["final_dq_norm"],
                "approach_to_coarse_to_finisher_success": bool(coarse_final["success"]),
                "approach_to_coarse_to_finisher_dwell_success": bool(coarse_final["success"]),
                "approach_to_coarse_to_finisher_final_position_error": coarse_final["final_position_error"],
                "approach_to_coarse_to_finisher_final_orientation_error": coarse_final["final_orientation_error"],
                "approach_to_coarse_to_finisher_min_position_error": coarse_final["min_position_error"],
                "approach_to_coarse_to_finisher_final_action_magnitude": coarse_final["final_action_magnitude"],
                "approach_to_coarse_to_finisher_final_dq_norm": coarse_final["final_dq_norm"],
                "conditional_success": bool(conditional["success"]),
                "conditional_dwell_success": bool(conditional["success"]),
                "conditional_final_position_error": conditional["final_position_error"],
                "conditional_final_orientation_error": conditional["final_orientation_error"],
                "conditional_min_position_error": conditional["min_position_error"],
                "conditional_final_action_magnitude": conditional["final_action_magnitude"],
                "conditional_final_dq_norm": conditional["final_dq_norm"],
                "handoff_position_error": handoff_result["final_position_error"] if handoff_result else None,
                "handoff_orientation_error": handoff_result["final_orientation_error"] if handoff_result else None,
                "handoff_action_magnitude": handoff_result["final_action_magnitude"] if handoff_result else None,
                "handoff_dq_norm": handoff_result["final_dq_norm"] if handoff_result else None,
                "coarse_final_position_error": coarse_result["final_position_error"] if coarse_result else None,
                "coarse_final_orientation_error": coarse_result["final_orientation_error"] if coarse_result else None,
                "coarse_final_action_magnitude": coarse_result["final_action_magnitude"] if coarse_result else None,
                "coarse_final_dq_norm": coarse_result["final_dq_norm"] if coarse_result else None,
            }
        )

    handoff_steps = [m["first_handoff_step"] for m in episode_metrics if m["first_handoff_step"] is not None]
    switching_summary = {
        "handoff_count": int(sum(m["handoff_count"] for m in episode_metrics)),
        "handoff_rate": _mean([m["handoff_count"] > 0 for m in episode_metrics]),
        "first_handoff_step_mean": float(np.mean(handoff_steps)) if handoff_steps else None,
        "handoff_ready_at_switch_rate": _mean([m["handoff_ready_at_switch"] for m in episode_metrics]),
        "success_after_handoff_rate_approach_to_finisher": _mean(
            [m["approach_to_finisher_success"] for m in episode_metrics if m["handoff_count"] > 0]
        ),
        "success_after_handoff_rate_approach_to_coarse_to_finisher": _mean(
            [m["approach_to_coarse_to_finisher_success"] for m in episode_metrics if m["handoff_count"] > 0]
        ),
        "success_after_handoff_rate_conditional": _mean(
            [m["conditional_success"] for m in episode_metrics if m["handoff_count"] > 0]
        ),
        "conditional_direct_route_rate": _mean([m["conditional_route"] == "direct_finisher" for m in episode_metrics]),
        "conditional_coarse_route_rate": _mean([m["conditional_route"] == "dock_coarse_then_finisher" for m in episode_metrics]),
    }
    pipeline_table = {
        "approach_only": _pipeline_summary(episode_metrics, "approach_only"),
        "approach_to_finisher": _pipeline_summary(episode_metrics, "approach_to_finisher"),
        "approach_to_dock_coarse_to_finisher": _pipeline_summary(episode_metrics, "approach_to_coarse_to_finisher"),
        "conditional_dock_coarse_fallback": _pipeline_summary(episode_metrics, "conditional"),
    }
    state_delta_summary = {
        "episode_count": len(delta_metrics),
        "delta_position_error_after_coarse_mean": _mean([m["delta_position_error_after_coarse"] for m in delta_metrics]),
        "delta_orientation_error_after_coarse_mean": _mean([m["delta_orientation_error_after_coarse"] for m in delta_metrics]),
        "delta_action_magnitude_after_coarse_mean": _mean([m["delta_action_magnitude_after_coarse"] for m in delta_metrics]),
        "delta_dq_norm_after_coarse_mean": _mean([m["delta_dq_norm_after_coarse"] for m in delta_metrics]),
        "position_error_improved_by_coarse_rate": _mean([m["delta_position_error_after_coarse"] < 0.0 for m in delta_metrics]),
        "orientation_error_improved_by_coarse_rate": _mean([m["delta_orientation_error_after_coarse"] < 0.0 for m in delta_metrics]),
        "action_magnitude_reduced_by_coarse_rate": _mean([m["delta_action_magnitude_after_coarse"] < 0.0 for m in delta_metrics]),
        "dq_norm_reduced_by_coarse_rate": _mean([m["delta_dq_norm_after_coarse"] < 0.0 for m in delta_metrics]),
        "handoff_ready_preserved_rate": _mean([m["handoff_ready_preserved"] for m in delta_metrics]),
        "handoff_ready_destroyed_rate": _mean([m["handoff_ready_destroyed"] for m in delta_metrics]),
        "strict_like_preserved_rate": _mean([m["strict_like_preserved"] for m in delta_metrics]),
        "strict_like_destroyed_rate": _mean([m["strict_like_destroyed"] for m in delta_metrics]),
        "episode_metrics": delta_metrics,
    }
    summary = {
        "approach_checkpoint": str(approach_checkpoint),
        "dock_coarse_checkpoint": str(dock_coarse_checkpoint),
        "finisher_checkpoint": str(finisher_checkpoint),
        "episodes": int(episodes),
        "seed": int(seed),
        "eval_scope": scope,
        "curriculum_stage_index": int(stage_index),
        "handoff_confirm_steps": int(handoff_confirm_steps),
        "handoff_mode": handoff_mode,
        "finisher_direct_thresholds": {
            "position_error": finisher_direct_pos_threshold,
            "orientation_error": finisher_direct_ori_threshold,
            "action_magnitude": finisher_direct_action_threshold,
            "dq_norm": finisher_direct_dq_threshold,
        },
        "strict_like_thresholds": {
            "position_error": strict_like_pos_threshold,
            "orientation_error": strict_like_ori_threshold,
            "action_magnitude": strict_like_action_threshold,
            "dq_norm": strict_like_dq_threshold,
        },
        "switching": switching_summary,
        "pipelines": pipeline_table,
        "episode_metrics": episode_metrics,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "pipeline_ablation_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "pipeline_ablation_summary.json", summary)
    write_json(artifact_root / "coarse_state_delta_summary.json", state_delta_summary)
    report = _build_markdown_report(summary, state_delta_summary)
    (artifact_root / "PIPELINE_ABLATION_REPORT.md").write_text(report)
    return summary


def _build_markdown_report(summary: dict[str, Any], delta_summary: dict[str, Any]) -> str:
    pipelines = summary["pipelines"]
    downstream_names = [
        "approach_to_finisher",
        "approach_to_dock_coarse_to_finisher",
        "conditional_dock_coarse_fallback",
    ]
    best_name = max(downstream_names, key=lambda name: pipelines[name]["overall_success_rate"])
    rows = []
    for name, metrics in pipelines.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{metrics['overall_success_rate']:.3f}",
                    f"{metrics['final_position_error_mean']:.5f}",
                    f"{metrics['final_orientation_error_mean']:.5f}",
                    f"{metrics['final_action_magnitude_mean']:.5f}",
                    f"{metrics['final_dq_norm_mean']:.5f}",
                ]
            )
            + " |"
        )
    if best_name == "approach_to_finisher":
        conclusion = "Approach -> Finisher is currently the strongest route, so Dock-Coarse should be skipped for this handoff distribution."
    elif best_name == "approach_to_dock_coarse_to_finisher":
        conclusion = "Approach -> Dock-Coarse -> Finisher is currently the strongest route, so Dock-Coarse remains useful as a fixed middle stage."
    elif best_name == "conditional_dock_coarse_fallback":
        conclusion = "The conditional route is currently strongest, so Dock-Coarse is better treated as a fallback than as a mandatory stage."
    else:  # pragma: no cover - defensive fallback for future pipeline names
        conclusion = "No downstream route clearly dominates under the current comparison."
    return "\n".join(
        [
            "# Pipeline Ablation Report",
            "",
            "Purpose: compare whether Dock-Coarse is still needed after the new Approach policy produces dock-coarse-ready handoff states.",
            f"Handoff mode: `{summary.get('handoff_mode', 'first_confirmed')}`.",
            "",
            "## Pipeline Comparison",
            "",
            "| Pipeline | Success | Final Pos Err | Final Ori Err | Final Action | Final DQ |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Dock-Coarse State Delta",
            "",
            f"- Mean position error delta after Dock-Coarse: {delta_summary['delta_position_error_after_coarse_mean']:.6f}",
            f"- Mean orientation error delta after Dock-Coarse: {delta_summary['delta_orientation_error_after_coarse_mean']:.6f}",
            f"- Mean action magnitude delta after Dock-Coarse: {delta_summary['delta_action_magnitude_after_coarse_mean']:.6f}",
            f"- Mean dq norm delta after Dock-Coarse: {delta_summary['delta_dq_norm_after_coarse_mean']:.6f}",
            f"- Handoff-ready preserved rate: {delta_summary['handoff_ready_preserved_rate']:.3f}",
            f"- Handoff-ready destroyed rate: {delta_summary['handoff_ready_destroyed_rate']:.3f}",
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


def _load_dock_coarse_config(config_path: str | None) -> dict[str, Any]:
    cfg = load_yaml_file(dock_coarse_default_config_path())
    if config_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(config_path)))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Approach/Dock-Coarse/Finisher pipeline ablations.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--dock-coarse-checkpoint", required=True)
    parser.add_argument("--finisher-checkpoint", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--approach-config")
    parser.add_argument("--dock-coarse-config")
    parser.add_argument("--finisher-config", required=True)
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--dock-coarse-algorithm", default="ppo")
    parser.add_argument("--finisher-algorithm", default="td3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--handoff-confirm-steps", type=int, default=2)
    parser.add_argument("--handoff-mode", choices=("first_confirmed", "final_settled"), default="first_confirmed")
    parser.add_argument("--finisher-direct-pos-threshold", type=float, default=0.005)
    parser.add_argument("--finisher-direct-ori-threshold", type=float, default=0.05)
    parser.add_argument("--finisher-direct-action-threshold", type=float, default=0.10)
    parser.add_argument("--finisher-direct-dq-threshold", type=float, default=0.002)
    parser.add_argument("--strict-like-pos-threshold", type=float, default=0.005)
    parser.add_argument("--strict-like-ori-threshold", type=float, default=0.05)
    parser.add_argument("--strict-like-action-threshold", type=float, default=0.10)
    parser.add_argument("--strict-like-dq-threshold", type=float, default=0.002)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    summary = evaluate_pipeline_ablation(
        approach_checkpoint=Path(args.approach_checkpoint),
        approach_algorithm=args.approach_algorithm,
        dock_coarse_checkpoint=Path(args.dock_coarse_checkpoint),
        dock_coarse_algorithm=args.dock_coarse_algorithm,
        finisher_checkpoint=Path(args.finisher_checkpoint),
        finisher_algorithm=args.finisher_algorithm,
        artifact_root=Path(args.artifact_root),
        approach_env_config=to_env_config(_load_approach_config(args.approach_config)),
        dock_coarse_env_config=to_env_config(_load_dock_coarse_config(args.dock_coarse_config)),
        finisher_env_config=to_env_config(load_yaml_file(Path(args.finisher_config))),
        episodes=args.episodes,
        seed=args.seed,
        stage_index=args.stage_index,
        handoff_confirm_steps=args.handoff_confirm_steps,
        handoff_mode=args.handoff_mode,
        finisher_direct_pos_threshold=args.finisher_direct_pos_threshold,
        finisher_direct_ori_threshold=args.finisher_direct_ori_threshold,
        finisher_direct_action_threshold=args.finisher_direct_action_threshold,
        finisher_direct_dq_threshold=args.finisher_direct_dq_threshold,
        strict_like_pos_threshold=args.strict_like_pos_threshold,
        strict_like_ori_threshold=args.strict_like_ori_threshold,
        strict_like_action_threshold=args.strict_like_action_threshold,
        strict_like_dq_threshold=args.strict_like_dq_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
