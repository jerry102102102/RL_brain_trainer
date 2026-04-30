"""Dock-Coarse and Dock-Coarse -> strict finisher evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..eval.eval_deterministic import _load_sb3_model
from ..eval.fixed_eval_suite import EvalEpisodeSpec, build_dock_eval_suite, suite_to_jsonable
from ..training.policy_config import (
    deep_merge,
    dock_coarse_default_config_path,
    load_yaml_file,
    to_env_config,
    write_json,
)


def _predict(model: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=float)


def _episode_reset_options(episode: EvalEpisodeSpec, *, policy_mode: str) -> dict[str, Any]:
    return {**episode.reset_options(), "policy_mode": policy_mode}


def _state_reset_options(result: dict[str, Any], *, policy_mode: str) -> dict[str, Any]:
    return {
        "initial_q": result["final_q"],
        "initial_dq": result["final_dq"],
        "initial_prev_action": result["final_prev_action"],
        "goal_q": result["goal_q"],
        "goal_pose6": result["goal_pose6"],
        "policy_mode": policy_mode,
    }


def _rollout(
    *,
    env: ArmKinematicEnv,
    model: Any,
    reset_options: dict[str, Any],
    coarse_reference_config: Phase1EnvConfig,
) -> dict[str, Any]:
    obs, info = env.reset(options=reset_options)
    initial_pos = float(info["position_error_norm"])
    initial_ori = float(info["orientation_error_norm"])
    actions: list[float] = []
    dq_norms: list[float] = []
    handoff_ready_actions: list[float] = []
    handoff_ready_dq_norms: list[float] = []
    min_pos = initial_pos
    min_ori = initial_ori
    coarse_cfg = coarse_reference_config.dock_coarse_reward_config
    last_pos = initial_pos
    last_ori = initial_ori
    last_action_magnitude = 0.0
    prev_in_coarse = initial_pos <= coarse_cfg.coarse_pos_threshold_m and initial_ori <= coarse_cfg.coarse_ori_threshold_rad
    prev_in_handoff_ready = (
        initial_pos <= coarse_cfg.handoff_ready_pos_threshold_m
        and initial_ori <= coarse_cfg.handoff_ready_ori_threshold_rad
        and float(np.linalg.norm(info["dq"])) <= coarse_cfg.handoff_ready_dq_threshold
        and float(np.linalg.norm(obs["prev_action"])) <= coarse_cfg.handoff_ready_action_threshold
    )
    position_only_coarse_hit = min_pos <= coarse_cfg.coarse_pos_threshold_m
    orientation_only_coarse_hit = min_ori <= coarse_cfg.coarse_ori_threshold_rad
    coarse_basin_hit = position_only_coarse_hit and orientation_only_coarse_hit
    handoff_ready_hit = prev_in_handoff_ready
    strict_like_hit = (
        min_pos <= coarse_cfg.strict_like_pos_threshold_m
        and min_ori <= coarse_cfg.strict_like_ori_threshold_rad
    )
    leave_coarse_basin_count = 0
    leave_handoff_ready_count = 0
    handoff_ready_dwell_count = 1 if prev_in_handoff_ready else 0
    max_handoff_ready_dwell_count = handoff_ready_dwell_count
    first_handoff_ready_step: int | None = 0 if prev_in_handoff_ready else None
    if prev_in_handoff_ready:
        handoff_ready_actions.append(float(np.linalg.norm(obs["prev_action"])))
        handoff_ready_dq_norms.append(float(np.linalg.norm(info["dq"])))
    position_regression_steps = 0
    orientation_regression_steps = 0
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = _predict(model, obs)
        last_action_magnitude = float(np.linalg.norm(action))
        actions.append(last_action_magnitude)
        obs, _, terminated, truncated, info = env.step(action)
        curr_dq_norm = float(np.linalg.norm(info["dq"]))
        dq_norms.append(curr_dq_norm)
        curr_pos = float(info["position_error_norm"])
        curr_ori = float(info["orientation_error_norm"])
        min_pos = min(min_pos, curr_pos)
        min_ori = min(min_ori, curr_ori)
        curr_in_coarse = curr_pos <= coarse_cfg.coarse_pos_threshold_m and curr_ori <= coarse_cfg.coarse_ori_threshold_rad
        if prev_in_coarse and not curr_in_coarse:
            leave_coarse_basin_count += 1
        if curr_pos > last_pos:
            position_regression_steps += 1
        if curr_ori > last_ori:
            orientation_regression_steps += 1
        position_only_coarse_hit = position_only_coarse_hit or curr_pos <= coarse_cfg.coarse_pos_threshold_m
        orientation_only_coarse_hit = orientation_only_coarse_hit or curr_ori <= coarse_cfg.coarse_ori_threshold_rad
        coarse_basin_hit = coarse_basin_hit or curr_in_coarse
        curr_in_handoff_ready = (
            curr_pos <= coarse_cfg.handoff_ready_pos_threshold_m
            and curr_ori <= coarse_cfg.handoff_ready_ori_threshold_rad
            and curr_dq_norm <= coarse_cfg.handoff_ready_dq_threshold
            and last_action_magnitude <= coarse_cfg.handoff_ready_action_threshold
        )
        handoff_ready_hit = handoff_ready_hit or curr_in_handoff_ready
        if curr_in_handoff_ready:
            if first_handoff_ready_step is None:
                first_handoff_ready_step = steps + 1
            handoff_ready_actions.append(last_action_magnitude)
            handoff_ready_dq_norms.append(curr_dq_norm)
        if prev_in_handoff_ready and not curr_in_handoff_ready:
            leave_handoff_ready_count += 1
        handoff_ready_dwell_count = handoff_ready_dwell_count + 1 if curr_in_handoff_ready else 0
        max_handoff_ready_dwell_count = max(max_handoff_ready_dwell_count, handoff_ready_dwell_count)
        strict_like_hit = strict_like_hit or (
            curr_pos <= coarse_cfg.strict_like_pos_threshold_m
            and curr_ori <= coarse_cfg.strict_like_ori_threshold_rad
        )
        prev_in_coarse = curr_in_coarse
        prev_in_handoff_ready = curr_in_handoff_ready
        last_pos = curr_pos
        last_ori = curr_ori
        steps += 1

    final_pos = float(info["position_error_norm"])
    final_ori = float(info["orientation_error_norm"])
    return {
        "initial_position_error": initial_pos,
        "initial_orientation_error": initial_ori,
        "success": bool(info["success"]),
        "position_only_coarse_hit": bool(position_only_coarse_hit),
        "orientation_only_coarse_hit": bool(orientation_only_coarse_hit),
        "coarse_basin_hit": bool(coarse_basin_hit),
        "handoff_ready_hit": bool(handoff_ready_hit),
        "handoff_ready_dwell": bool(max_handoff_ready_dwell_count >= coarse_cfg.coarse_dwell_start),
        "max_handoff_ready_dwell_count": int(max_handoff_ready_dwell_count),
        "first_handoff_ready_step": int(first_handoff_ready_step) if first_handoff_ready_step is not None else None,
        "strict_like_hit": bool(strict_like_hit),
        "dwell_success": int(info["dwell_count"]) >= env.config.dwell_steps_target,
        "final_position_error": final_pos,
        "final_orientation_error": final_ori,
        "min_position_error": min_pos,
        "min_orientation_error": min_ori,
        "final_minus_min_position_error": final_pos - min_pos,
        "final_minus_min_orientation_error": final_ori - min_ori,
        "regression": bool(final_pos > min_pos + 0.005),
        "leave_coarse_basin": bool(leave_coarse_basin_count > 0),
        "leave_coarse_basin_count": int(leave_coarse_basin_count),
        "leave_handoff_ready": bool(leave_handoff_ready_count > 0),
        "leave_handoff_ready_count": int(leave_handoff_ready_count),
        "position_regression_rate": float(position_regression_steps / max(steps, 1)),
        "orientation_regression_rate": float(orientation_regression_steps / max(steps, 1)),
        "leave_zone": bool(
            final_pos > coarse_cfg.working_pos_radius_m
            or final_ori > coarse_cfg.working_ori_radius_rad
        ),
        "mean_action_magnitude": float(np.mean(actions)) if actions else 0.0,
        "final_action_magnitude": float(last_action_magnitude),
        "mean_dq_norm": float(np.mean(dq_norms)) if dq_norms else 0.0,
        "mean_action_magnitude_inside_handoff_ready": float(np.mean(handoff_ready_actions))
        if handoff_ready_actions
        else 0.0,
        "mean_dq_norm_inside_handoff_ready": float(np.mean(handoff_ready_dq_norms))
        if handoff_ready_dq_norms
        else 0.0,
        "final_dq_norm": float(np.linalg.norm(info["dq"])),
        "step_count": int(steps),
        "final_q": np.asarray(info["q"], dtype=float).tolist(),
        "final_dq": np.asarray(info["dq"], dtype=float).tolist(),
        "final_prev_action": np.asarray(env._prev_action, dtype=float).tolist(),  # noqa: SLF001 - handoff eval state
        "goal_q": np.asarray(info["goal_q"], dtype=float).tolist(),
        "goal_pose6": np.asarray(info["goal_pose6"], dtype=float).tolist(),
    }


def _bucket(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels, strict=False):
        if value <= edge:
            return label
    return labels[-1]


def _bucket_summary(metrics: list[dict[str, Any]], *, value_key: str, prefix: str) -> dict[str, dict[str, float | int]]:
    if value_key == "initial_position_error":
        labels = ["<=0.5cm", "0.5-1.0cm", "1.0-2.0cm", ">2.0cm"]
        edges = [0.005, 0.010, 0.020]
    else:
        labels = ["<=0.05rad", "0.05-0.10rad", "0.10-0.20rad", "0.20-0.30rad", ">0.30rad"]
        edges = [0.05, 0.10, 0.20, 0.30]

    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for metric in metrics:
        grouped[_bucket(float(metric[value_key]), edges, labels)].append(metric)

    summary: dict[str, dict[str, float | int]] = {}
    for label, items in grouped.items():
        if not items:
            summary[label] = {"count": 0, "success_rate": 0.0, "coarse_basin_entry_rate": 0.0}
            continue
        summary[label] = {
            "count": len(items),
            "success_rate": float(np.mean([bool(item["success"]) for item in items])),
            "coarse_basin_entry_rate": float(np.mean([bool(item["coarse_basin_hit"]) for item in items])),
            "handoff_ready_hit_rate": float(np.mean([bool(item["handoff_ready_hit"]) for item in items])),
            "handoff_ready_dwell_rate": float(np.mean([bool(item["handoff_ready_dwell"]) for item in items])),
            "strict_like_hit_rate": float(np.mean([bool(item["strict_like_hit"]) for item in items])),
            f"mean_{prefix}": float(np.mean([float(item[value_key]) for item in items])),
        }
    return summary


def _summary(prefix: str, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {f"{prefix}_episode_count": 0}
    return {
        f"{prefix}_episode_count": len(metrics),
        f"{prefix}_success_rate": float(np.mean([bool(m["success"]) for m in metrics])),
        f"{prefix}_coarse_basin_entry_rate": float(np.mean([bool(m["coarse_basin_hit"]) for m in metrics])),
        f"{prefix}_handoff_ready_hit_rate": float(np.mean([bool(m["handoff_ready_hit"]) for m in metrics])),
        f"{prefix}_handoff_ready_dwell_rate": float(np.mean([bool(m["handoff_ready_dwell"]) for m in metrics])),
        f"{prefix}_mean_max_handoff_ready_dwell_count": float(
            np.mean([float(m["max_handoff_ready_dwell_count"]) for m in metrics])
        ),
        f"{prefix}_max_handoff_ready_dwell_count": int(
            max([int(m["max_handoff_ready_dwell_count"]) for m in metrics])
        ),
        f"{prefix}_mean_first_handoff_ready_step": float(
            np.mean([float(m["first_handoff_ready_step"]) for m in metrics if m["first_handoff_ready_step"] is not None])
        )
        if any(m["first_handoff_ready_step"] is not None for m in metrics)
        else None,
        f"{prefix}_position_only_coarse_hit_rate": float(
            np.mean([bool(m["position_only_coarse_hit"]) for m in metrics])
        ),
        f"{prefix}_orientation_only_coarse_hit_rate": float(
            np.mean([bool(m["orientation_only_coarse_hit"]) for m in metrics])
        ),
        f"{prefix}_strict_like_hit_rate": float(np.mean([bool(m["strict_like_hit"]) for m in metrics])),
        f"{prefix}_dwell_success_rate": float(np.mean([bool(m["dwell_success"]) for m in metrics])),
        f"{prefix}_mean_initial_position_error": float(np.mean([float(m["initial_position_error"]) for m in metrics])),
        f"{prefix}_mean_initial_orientation_error": float(np.mean([float(m["initial_orientation_error"]) for m in metrics])),
        f"{prefix}_mean_final_position_error": float(np.mean([float(m["final_position_error"]) for m in metrics])),
        f"{prefix}_mean_final_orientation_error": float(np.mean([float(m["final_orientation_error"]) for m in metrics])),
        f"{prefix}_mean_min_position_error": float(np.mean([float(m["min_position_error"]) for m in metrics])),
        f"{prefix}_mean_min_orientation_error": float(np.mean([float(m["min_orientation_error"]) for m in metrics])),
        f"{prefix}_mean_final_minus_min_position_error": float(
            np.mean([float(m["final_minus_min_position_error"]) for m in metrics])
        ),
        f"{prefix}_mean_final_minus_min_orientation_error": float(
            np.mean([float(m["final_minus_min_orientation_error"]) for m in metrics])
        ),
        f"{prefix}_mean_final_dq_norm": float(np.mean([float(m["final_dq_norm"]) for m in metrics])),
        f"{prefix}_mean_final_action_magnitude": float(
            np.mean([float(m["final_action_magnitude"]) for m in metrics])
        ),
        f"{prefix}_mean_action_magnitude": float(np.mean([float(m["mean_action_magnitude"]) for m in metrics])),
        f"{prefix}_mean_action_magnitude_inside_handoff_ready": float(
            np.mean([float(m["mean_action_magnitude_inside_handoff_ready"]) for m in metrics])
        ),
        f"{prefix}_mean_dq_norm_inside_handoff_ready": float(
            np.mean([float(m["mean_dq_norm_inside_handoff_ready"]) for m in metrics])
        ),
        f"{prefix}_regression_rate": float(np.mean([bool(m["regression"]) for m in metrics])),
        f"{prefix}_leave_coarse_basin_rate": float(np.mean([bool(m["leave_coarse_basin"]) for m in metrics])),
        f"{prefix}_mean_leave_coarse_basin_count": float(
            np.mean([float(m["leave_coarse_basin_count"]) for m in metrics])
        ),
        f"{prefix}_leave_handoff_ready_rate": float(np.mean([bool(m["leave_handoff_ready"]) for m in metrics])),
        f"{prefix}_mean_leave_handoff_ready_count": float(
            np.mean([float(m["leave_handoff_ready_count"]) for m in metrics])
        ),
        f"{prefix}_position_regression_rate": float(
            np.mean([float(m["position_regression_rate"]) for m in metrics])
        ),
        f"{prefix}_orientation_regression_rate": float(
            np.mean([float(m["orientation_regression_rate"]) for m in metrics])
        ),
        f"{prefix}_leave_zone_rate": float(np.mean([bool(m["leave_zone"]) for m in metrics])),
        f"{prefix}_initial_position_bucket_summary": _bucket_summary(
            metrics,
            value_key="initial_position_error",
            prefix="initial_position_error",
        ),
        f"{prefix}_initial_orientation_bucket_summary": _bucket_summary(
            metrics,
            value_key="initial_orientation_error",
            prefix="initial_orientation_error",
        ),
    }


def evaluate_dock_coarse_policy(
    *,
    coarse_checkpoint: Path,
    coarse_algorithm: str,
    artifact_root: Path,
    env_config: Phase1EnvConfig,
    episodes: int = 50,
    seed: int = 700001,
    finisher_checkpoint: Path | None = None,
    finisher_algorithm: str = "td3",
    finisher_env_config: Phase1EnvConfig | None = None,
) -> dict[str, Any]:
    coarse_model = _load_sb3_model(coarse_algorithm, coarse_checkpoint)
    suite = build_dock_eval_suite(env_config, seed=seed, n_episodes=episodes)
    coarse_metrics: list[dict[str, Any]] = []
    direct_finisher_metrics: list[dict[str, Any]] = []
    coarse_then_finisher_metrics: list[dict[str, Any]] = []

    for episode in suite:
        env = ArmKinematicEnv(config=env_config)
        env.set_policy_mode("dock_coarse")
        coarse_metrics.append(
            _rollout(
                env=env,
                model=coarse_model,
                reset_options=_episode_reset_options(episode, policy_mode="dock_coarse"),
                coarse_reference_config=env_config,
            )
        )

    if finisher_checkpoint is not None:
        finisher_model = _load_sb3_model(finisher_algorithm, finisher_checkpoint)
        finisher_config = finisher_env_config or env_config
        for episode, coarse_result in zip(suite, coarse_metrics, strict=True):
            direct_env = ArmKinematicEnv(config=finisher_config)
            direct_env.set_policy_mode("dock")
            direct_finisher_metrics.append(
                _rollout(
                    env=direct_env,
                    model=finisher_model,
                    reset_options=_episode_reset_options(episode, policy_mode="dock"),
                    coarse_reference_config=env_config,
                )
            )

            finisher_env = ArmKinematicEnv(config=finisher_config)
            finisher_env.set_policy_mode("dock")
            coarse_then_finisher_metrics.append(
                _rollout(
                    env=finisher_env,
                    model=finisher_model,
                    reset_options=_state_reset_options(coarse_result, policy_mode="dock"),
                    coarse_reference_config=env_config,
                )
            )

    summary: dict[str, Any] = {
        "coarse_checkpoint": str(coarse_checkpoint),
        "coarse_algorithm": coarse_algorithm,
        "finisher_checkpoint": str(finisher_checkpoint) if finisher_checkpoint else None,
        "finisher_algorithm": finisher_algorithm if finisher_checkpoint else None,
        "episodes": int(episodes),
        "seed": int(seed),
        "coarse_success_definition": {
            "position_threshold_m": env_config.dock_coarse_reward_config.coarse_pos_threshold_m,
            "orientation_threshold_rad": env_config.dock_coarse_reward_config.coarse_ori_threshold_rad,
            "dwell_steps": env_config.dwell_steps_target,
        },
        **_summary("coarse", coarse_metrics),
    }
    if direct_finisher_metrics:
        summary.update(_summary("direct_finisher", direct_finisher_metrics))
        summary.update(_summary("coarse_then_finisher", coarse_then_finisher_metrics))
        summary["coarse_then_finisher_success_delta_vs_direct"] = float(
            summary["coarse_then_finisher_success_rate"] - summary["direct_finisher_success_rate"]
        )
        summary["coarse_then_finisher_strict_like_delta_vs_direct"] = float(
            summary["coarse_then_finisher_strict_like_hit_rate"] - summary["direct_finisher_strict_like_hit_rate"]
        )

    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "dock_coarse_eval_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "dock_coarse_eval_summary.json", summary)
    write_json(
        artifact_root / "dock_coarse_eval_episodes.json",
        {
            "coarse_metrics": coarse_metrics,
            "direct_finisher_metrics": direct_finisher_metrics,
            "coarse_then_finisher_metrics": coarse_then_finisher_metrics,
        },
    )
    return summary


def _load_config(config_path: str | None) -> dict[str, Any]:
    cfg = load_yaml_file(dock_coarse_default_config_path())
    if config_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(config_path)))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 1C Dock-Coarse policy.")
    parser.add_argument("--coarse-checkpoint", required=True)
    parser.add_argument("--coarse-algorithm", default="ppo", choices=("ppo", "td3"))
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--finisher-checkpoint")
    parser.add_argument("--finisher-algorithm", default="td3", choices=("ppo", "td3"))
    parser.add_argument("--finisher-config")
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    env_config = to_env_config(_load_config(args.config))
    finisher_env_config = to_env_config(load_yaml_file(Path(args.finisher_config))) if args.finisher_config else None
    summary = evaluate_dock_coarse_policy(
        coarse_checkpoint=Path(args.coarse_checkpoint),
        coarse_algorithm=args.coarse_algorithm,
        artifact_root=Path(args.artifact_root),
        env_config=env_config,
        episodes=args.episodes,
        seed=args.seed,
        finisher_checkpoint=Path(args.finisher_checkpoint) if args.finisher_checkpoint else None,
        finisher_algorithm=args.finisher_algorithm,
        finisher_env_config=finisher_env_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
