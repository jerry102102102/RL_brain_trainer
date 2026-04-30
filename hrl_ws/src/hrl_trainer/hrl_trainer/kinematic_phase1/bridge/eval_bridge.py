"""Bridge-only and Bridge -> Dock evaluation for Phase 1C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..eval.eval_deterministic import _load_sb3_model
from ..handoff.handoff_dataset import to_jsonable
from ..training.policy_config import (
    bridge_default_config_path,
    deep_merge,
    load_yaml_file,
    to_env_config,
    write_json,
)
from .bridge_reset_samplers import load_bridge_handoff_states, sample_bridge_reset


def _predict(model: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=float)


def _sample_bridge_reset_options(env_config: Phase1EnvConfig, *, seed: int, episodes: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records = load_bridge_handoff_states(env_config.bridge_reset_config)
    return [
        sample_bridge_reset(
            rng=rng,
            records=records,
            joint_specs=env_config.joint_specs,
            config=env_config.bridge_reset_config,
        )
        for _ in range(episodes)
    ]


def _episode_reset_options(reset_options: dict[str, Any], *, policy_mode: str) -> dict[str, Any]:
    return {
        "initial_q": reset_options["initial_q"],
        "initial_dq": reset_options["initial_dq"],
        "initial_prev_action": reset_options["initial_prev_action"],
        "goal_q": reset_options["goal_q"],
        "goal_pose6": reset_options["goal_pose6"],
        "policy_mode": policy_mode,
    }


def _in_bridge_basin(info: dict[str, Any], env_config: Phase1EnvConfig) -> bool:
    return (
        float(info["position_error_norm"]) <= env_config.bridge_reward_config.acceptance_pos_threshold_m
        and float(info["orientation_error_norm"]) <= env_config.bridge_reward_config.acceptance_ori_threshold_rad
    )


def _rollout(
    *,
    env: ArmKinematicEnv,
    model: Any,
    reset_options: dict[str, Any],
    policy_mode: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    obs, info = env.reset(options=_episode_reset_options(reset_options, policy_mode=policy_mode))
    initial_info = dict(info)
    actions: list[float] = []
    dq_norms: list[float] = []
    min_pos = float(info["position_error_norm"])
    min_ori = float(info["orientation_error_norm"])
    max_pos = float(info["position_error_norm"])
    basin_hit = _in_bridge_basin(info, env.config)
    coarse_orientation_hit = float(info["orientation_error_norm"]) <= env.config.bridge_reward_config.coarse_orientation_threshold_rad
    return_position_hit = (
        coarse_orientation_hit
        and float(info["position_error_norm"]) <= env.config.bridge_reward_config.acceptance_pos_threshold_m
    )
    terminated = False
    truncated = False
    steps = 0
    horizon = max_steps if max_steps is not None else env.config.episode_length
    while not (terminated or truncated) and steps < horizon:
        action = _predict(model, obs)
        actions.append(float(np.linalg.norm(action)))
        obs, _, terminated, truncated, info = env.step(action)
        dq_norms.append(float(np.linalg.norm(info["dq"])))
        min_pos = min(min_pos, float(info["position_error_norm"]))
        min_ori = min(min_ori, float(info["orientation_error_norm"]))
        max_pos = max(max_pos, float(info["position_error_norm"]))
        basin_hit = basin_hit or _in_bridge_basin(info, env.config)
        coarse_orientation_hit = (
            coarse_orientation_hit
            or float(info["orientation_error_norm"]) <= env.config.bridge_reward_config.coarse_orientation_threshold_rad
        )
        return_position_hit = return_position_hit or (
            coarse_orientation_hit
            and float(info["position_error_norm"]) <= env.config.bridge_reward_config.acceptance_pos_threshold_m
        )
        steps += 1
    final_pos = float(info["position_error_norm"])
    final_ori = float(info["orientation_error_norm"])
    return {
        "initial_position_error": float(initial_info["position_error_norm"]),
        "initial_orientation_error": float(initial_info["orientation_error_norm"]),
        "initial_dq_norm": float(np.linalg.norm(initial_info["dq"])),
        "source_bucket_type": reset_options.get("source_bucket_type", "unknown"),
        "success": bool(info["success"]),
        "basin_hit": bool(basin_hit),
        "final_position_error": final_pos,
        "final_orientation_error": final_ori,
        "min_position_error": min_pos,
        "min_orientation_error": min_ori,
        "max_position_error": max_pos,
        "final_minus_min_position_error": final_pos - min_pos,
        "final_minus_min_orientation_error": final_ori - min_ori,
        "coarse_orientation_hit": bool(coarse_orientation_hit),
        "return_position_hit": bool(return_position_hit),
        "final_dq_norm": float(np.linalg.norm(info["dq"])),
        "mean_action_magnitude": float(np.mean(actions)) if actions else 0.0,
        "mean_dq_norm": float(np.mean(dq_norms)) if dq_norms else 0.0,
        "regression": bool(final_pos > min_pos + 0.003),
        "leave_near_goal": bool(final_pos > env.config.bridge_reward_config.position_keep_radius_m),
        "step_count": int(steps),
        "final_q": np.asarray(info["q"], dtype=float).tolist(),
        "final_dq": np.asarray(info["dq"], dtype=float).tolist(),
        "final_prev_action": np.asarray(env._prev_action, dtype=float).tolist(),  # noqa: SLF001 - evaluation handoff state
        "goal_q": np.asarray(info["goal_q"], dtype=float).tolist(),
        "goal_pose6": np.asarray(info["goal_pose6"], dtype=float).tolist(),
    }


def _summary(prefix: str, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {f"{prefix}_episode_count": 0}
    return {
        f"{prefix}_episode_count": len(metrics),
        f"{prefix}_success_rate": float(np.mean([bool(m["success"]) for m in metrics])),
        f"{prefix}_basin_entry_rate": float(np.mean([bool(m.get("basin_hit", False)) for m in metrics])),
        f"{prefix}_mean_initial_position_error": float(np.mean([float(m["initial_position_error"]) for m in metrics])),
        f"{prefix}_mean_initial_orientation_error": float(np.mean([float(m["initial_orientation_error"]) for m in metrics])),
        f"{prefix}_mean_final_position_error": float(np.mean([float(m["final_position_error"]) for m in metrics])),
        f"{prefix}_mean_final_orientation_error": float(np.mean([float(m["final_orientation_error"]) for m in metrics])),
        f"{prefix}_mean_min_position_error": float(np.mean([float(m["min_position_error"]) for m in metrics])),
        f"{prefix}_mean_min_orientation_error": float(np.mean([float(m["min_orientation_error"]) for m in metrics])),
        f"{prefix}_mean_max_position_error": float(np.mean([float(m["max_position_error"]) for m in metrics])),
        f"{prefix}_coarse_orientation_hit_rate": float(np.mean([bool(m["coarse_orientation_hit"]) for m in metrics])),
        f"{prefix}_return_position_hit_rate": float(np.mean([bool(m["return_position_hit"]) for m in metrics])),
        f"{prefix}_mean_final_minus_min_orientation_error": float(
            np.mean([float(m["final_minus_min_orientation_error"]) for m in metrics])
        ),
        f"{prefix}_mean_final_dq_norm": float(np.mean([float(m["final_dq_norm"]) for m in metrics])),
        f"{prefix}_mean_action_magnitude": float(np.mean([float(m["mean_action_magnitude"]) for m in metrics])),
        f"{prefix}_regression_rate": float(np.mean([bool(m["regression"]) for m in metrics])),
        f"{prefix}_leave_near_goal_rate": float(np.mean([bool(m["leave_near_goal"]) for m in metrics])),
    }


def evaluate_bridge_policy(
    *,
    bridge_checkpoint: Path,
    bridge_algorithm: str,
    artifact_root: Path,
    env_config: Phase1EnvConfig,
    episodes: int = 100,
    seed: int = 920001,
    dock_checkpoint: Path | None = None,
    dock_algorithm: str = "td3",
    dock_env_config: Phase1EnvConfig | None = None,
) -> dict[str, Any]:
    bridge_model = _load_sb3_model(bridge_algorithm, bridge_checkpoint)
    reset_suite = _sample_bridge_reset_options(env_config, seed=seed, episodes=episodes)
    bridge_metrics: list[dict[str, Any]] = []
    direct_dock_metrics: list[dict[str, Any]] = []
    bridge_then_dock_metrics: list[dict[str, Any]] = []

    for reset_options in reset_suite:
        bridge_env = ArmKinematicEnv(config=env_config)
        bridge_env.set_policy_mode("bridge")
        bridge_result = _rollout(env=bridge_env, model=bridge_model, reset_options=reset_options, policy_mode="bridge")
        bridge_metrics.append(bridge_result)

    if dock_checkpoint is not None:
        dock_model = _load_sb3_model(dock_algorithm, dock_checkpoint)
        dock_config = dock_env_config or env_config
        for reset_options, bridge_result in zip(reset_suite, bridge_metrics, strict=True):
            direct_env = ArmKinematicEnv(config=dock_config)
            direct_env.set_policy_mode("dock")
            direct_dock_metrics.append(
                _rollout(env=direct_env, model=dock_model, reset_options=reset_options, policy_mode="dock")
            )

            dock_start = {
                "initial_q": bridge_result["final_q"],
                "initial_dq": bridge_result["final_dq"],
                "initial_prev_action": bridge_result["final_prev_action"],
                "goal_q": bridge_result["goal_q"],
                "goal_pose6": bridge_result["goal_pose6"],
                "source_bucket_type": bridge_result.get("source_bucket_type", "unknown"),
            }
            dock_env = ArmKinematicEnv(config=dock_config)
            dock_env.set_policy_mode("dock")
            bridge_then_dock_metrics.append(
                _rollout(env=dock_env, model=dock_model, reset_options=dock_start, policy_mode="dock")
            )

    summary: dict[str, Any] = {
        "bridge_checkpoint": str(bridge_checkpoint),
        "bridge_algorithm": bridge_algorithm,
        "dock_checkpoint": str(dock_checkpoint) if dock_checkpoint else None,
        "dock_algorithm": dock_algorithm if dock_checkpoint else None,
        "episodes": int(episodes),
        "seed": int(seed),
        "bridge_success_definition": {
            "type": "geometry_basin_entry",
            "position_threshold_m": env_config.bridge_reward_config.acceptance_pos_threshold_m,
            "orientation_threshold_rad": env_config.bridge_reward_config.acceptance_ori_threshold_rad,
        },
        **_summary("bridge", bridge_metrics),
    }
    if direct_dock_metrics:
        summary.update(_summary("direct_dock", direct_dock_metrics))
        summary.update(_summary("bridge_then_dock", bridge_then_dock_metrics))
        summary["bridge_then_dock_success_delta_vs_direct"] = float(
            summary["bridge_then_dock_success_rate"] - summary["direct_dock_success_rate"]
        )
        summary["bridge_then_dock_basin_delta_vs_direct"] = float(
            summary["bridge_then_dock_basin_entry_rate"] - summary["direct_dock_basin_entry_rate"]
        )

    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "bridge_eval_summary.json", to_jsonable(summary))
    write_json(
        artifact_root / "bridge_eval_episodes.json",
        to_jsonable(
            {
                "bridge_metrics": bridge_metrics,
                "direct_dock_metrics": direct_dock_metrics,
                "bridge_then_dock_metrics": bridge_then_dock_metrics,
            }
        ),
    )
    return to_jsonable(summary)


def _load_config(config_path: str | None) -> dict[str, Any]:
    cfg = load_yaml_file(bridge_default_config_path())
    if config_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(config_path)))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 1C Bridge policy.")
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--bridge-algorithm", default="ppo", choices=("ppo", "td3"))
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=920001)
    parser.add_argument("--dock-checkpoint")
    parser.add_argument("--dock-algorithm", default="td3", choices=("ppo", "td3"))
    parser.add_argument("--dock-config")
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    env_config = to_env_config(_load_config(args.config))
    dock_env_config = to_env_config(load_yaml_file(Path(args.dock_config))) if args.dock_config else None
    summary = evaluate_bridge_policy(
        bridge_checkpoint=Path(args.bridge_checkpoint),
        bridge_algorithm=args.bridge_algorithm,
        artifact_root=Path(args.artifact_root),
        env_config=env_config,
        episodes=args.episodes,
        seed=args.seed,
        dock_checkpoint=Path(args.dock_checkpoint) if args.dock_checkpoint else None,
        dock_algorithm=args.dock_algorithm,
        dock_env_config=dock_env_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
