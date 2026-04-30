"""Hold-focused Approach-only evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..training.policy_config import approach_default_config_path, deep_merge, load_yaml_file, to_env_config, write_json
from .eval_deterministic import _load_sb3_model
from .fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable


def _mean(values: list[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _run_hold_eval(
    *,
    checkpoint: Path,
    algorithm: str,
    env_cfg,
    episodes: int,
    seed: int,
    stage_index: int,
    strict_pos_m: float,
    strict_ori_rad: float,
    action_threshold: float,
    dq_threshold: float,
) -> dict[str, Any]:
    model = _load_sb3_model(algorithm, checkpoint)
    if env_cfg.curriculum_config.enabled and env_cfg.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(env_cfg, seed=seed, stage_index=stage_index, n_episodes=episodes)
        scope = "curriculum_region"
    else:
        suite = build_fixed_eval_suite(seed=seed, n_episodes=episodes, joint_specs=env_cfg.joint_specs)
        scope = "fixed_random"

    metrics: list[dict[str, Any]] = []
    for episode in suite:
        env = ArmKinematicEnv(config=env_cfg)
        env.set_curriculum_stage(stage_index)
        env.set_policy_mode("approach")
        obs, info = env.reset(options={**episode.reset_options(), "policy_mode": "approach"})
        terminated = False
        truncated = False
        step = 0
        min_pos = float(info["position_error_norm"])
        min_ori = float(info["orientation_error_norm"])
        final_action = 0.0
        final_dq = 0.0
        strict_pose_streak = 0
        strict_hold_streak = 0
        max_strict_pose_streak = 0
        max_strict_hold_streak = 0
        strict_pose_hit = False
        strict_hold_hit = False
        first_strict_hold_step: int | None = None
        action_norms: list[float] = []
        dq_norms: list[float] = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=float)
            action_norm = float(np.linalg.norm(action))
            obs, _, terminated, truncated, info = env.step(action)
            step += 1
            dq_norm = float(info.get("executed_delta_q_l2", np.linalg.norm(info["dq"])))
            pos = float(info["position_error_norm"])
            ori = float(info["orientation_error_norm"])
            min_pos = min(min_pos, pos)
            min_ori = min(min_ori, ori)
            final_action = action_norm
            final_dq = dq_norm
            action_norms.append(action_norm)
            dq_norms.append(dq_norm)
            in_strict_pose = pos <= strict_pos_m and ori <= strict_ori_rad
            in_strict_hold = (
                in_strict_pose
                and (action_threshold <= 0.0 or action_norm <= action_threshold)
                and (dq_threshold <= 0.0 or dq_norm <= dq_threshold)
            )
            strict_pose_hit = strict_pose_hit or in_strict_pose
            strict_hold_hit = strict_hold_hit or in_strict_hold
            strict_pose_streak = strict_pose_streak + 1 if in_strict_pose else 0
            strict_hold_streak = strict_hold_streak + 1 if in_strict_hold else 0
            max_strict_pose_streak = max(max_strict_pose_streak, strict_pose_streak)
            max_strict_hold_streak = max(max_strict_hold_streak, strict_hold_streak)
            if in_strict_hold and first_strict_hold_step is None:
                first_strict_hold_step = step

        final_pos = float(info["position_error_norm"])
        final_ori = float(info["orientation_error_norm"])
        final_strict_pose = final_pos <= strict_pos_m and final_ori <= strict_ori_rad
        final_strict_hold = (
            final_strict_pose
            and (action_threshold <= 0.0 or final_action <= action_threshold)
            and (dq_threshold <= 0.0 or final_dq <= dq_threshold)
        )
        metrics.append(
            {
                "episode_id": int(episode.episode_id),
                "strict_pose_hit": bool(strict_pose_hit),
                "strict_hold_hit": bool(strict_hold_hit),
                "final_strict_pose": bool(final_strict_pose),
                "final_strict_hold": bool(final_strict_hold),
                "max_consecutive_strict_pose_steps": int(max_strict_pose_streak),
                "max_consecutive_strict_hold_steps": int(max_strict_hold_streak),
                "first_strict_hold_step": first_strict_hold_step,
                "final_position_error": final_pos,
                "final_orientation_error": final_ori,
                "min_position_error": min_pos,
                "min_orientation_error": min_ori,
                "final_action_magnitude": final_action,
                "final_dq_norm": final_dq,
                "mean_action_magnitude": _mean(action_norms),
                "mean_dq_norm": _mean(dq_norms),
            }
        )

    first_steps = [m["first_strict_hold_step"] for m in metrics if m["first_strict_hold_step"] is not None]
    return {
        "checkpoint": str(checkpoint),
        "eval_scope": scope,
        "episodes": int(episodes),
        "seed": int(seed),
        "strict_pos_threshold_m": float(strict_pos_m),
        "strict_ori_threshold_rad": float(strict_ori_rad),
        "action_threshold": float(action_threshold),
        "dq_threshold": float(dq_threshold),
        "strict_pose_hit_rate": _mean([m["strict_pose_hit"] for m in metrics]),
        "strict_hold_hit_rate": _mean([m["strict_hold_hit"] for m in metrics]),
        "final_strict_pose_rate": _mean([m["final_strict_pose"] for m in metrics]),
        "final_strict_hold_rate": _mean([m["final_strict_hold"] for m in metrics]),
        "max_consecutive_strict_pose_steps_mean": _mean([m["max_consecutive_strict_pose_steps"] for m in metrics]),
        "max_consecutive_strict_hold_steps_mean": _mean([m["max_consecutive_strict_hold_steps"] for m in metrics]),
        "max_consecutive_strict_hold_steps_max": int(max([m["max_consecutive_strict_hold_steps"] for m in metrics], default=0)),
        "mean_time_to_strict_hold": float(np.mean(first_steps)) if first_steps else None,
        "mean_final_position_error": _mean([m["final_position_error"] for m in metrics]),
        "mean_final_orientation_error": _mean([m["final_orientation_error"] for m in metrics]),
        "mean_min_position_error": _mean([m["min_position_error"] for m in metrics]),
        "mean_min_orientation_error": _mean([m["min_orientation_error"] for m in metrics]),
        "mean_final_action_magnitude": _mean([m["final_action_magnitude"] for m in metrics]),
        "mean_final_dq_norm": _mean([m["final_dq_norm"] for m in metrics]),
        "mean_action_magnitude": _mean([m["mean_action_magnitude"] for m in metrics]),
        "mean_dq_norm": _mean([m["mean_dq_norm"] for m in metrics]),
        "episode_metrics": metrics,
        "suite": suite_to_jsonable(suite),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate whether Approach alone can hold a docking-quality endpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--algorithm", default="ppo")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--strict-pos-m", type=float, default=0.001)
    parser.add_argument("--strict-ori-rad", type=float, default=0.005)
    parser.add_argument("--action-threshold", type=float, default=0.02)
    parser.add_argument("--dq-threshold", type=float, default=0.0005)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    cfg = deep_merge(load_yaml_file(approach_default_config_path()), load_yaml_file(Path(args.config)))
    summary = _run_hold_eval(
        checkpoint=Path(args.checkpoint),
        algorithm=args.algorithm,
        env_cfg=to_env_config(cfg),
        episodes=args.episodes,
        seed=args.seed,
        stage_index=args.stage_index,
        strict_pos_m=args.strict_pos_m,
        strict_ori_rad=args.strict_ori_rad,
        action_threshold=args.action_threshold,
        dq_threshold=args.dq_threshold,
    )
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    suite = summary.pop("suite")
    write_json(artifact_root / "approach_hold_eval_suite.json", {"suite": suite})
    write_json(artifact_root / "approach_hold_eval_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "episode_metrics"}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
