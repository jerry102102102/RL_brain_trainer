"""Route-curriculum numeric evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..route.reward_route import RouteRewardConfig, route_ready
from ..route.route_dataset import RouteDataset, load_route_dataset
from ..route.route_env import RouteEnvConfig, RouteKinematicEnv
from ..route.route_observation import RouteObservationConfig
from ..route.route_reset_samplers import RouteResetSamplerConfig
from ..training.policy_config import approach_default_config_path, deep_merge, load_yaml_file, ppo_default_config_path, to_env_config, write_json
from .eval_deterministic import _load_sb3_model


def _load_route_config(config_path: str | Path | None) -> dict[str, Any]:
    cfg = deep_merge(load_yaml_file(approach_default_config_path()), load_yaml_file(ppo_default_config_path()))
    if config_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(config_path)))
    return cfg


def _route_reward_config(cfg: dict[str, Any]) -> RouteRewardConfig:
    return RouteRewardConfig(**cfg.get("route", {}).get("reward", {}))


def _route_reset_config(cfg: dict[str, Any], *, max_route_index: int) -> RouteResetSamplerConfig:
    reset_cfg = dict(cfg.get("route", {}).get("reset", {}))
    reset_cfg.setdefault("max_route_index", max_route_index)
    return RouteResetSamplerConfig(**reset_cfg)


def _make_route_env(*, route: RouteDataset, cfg: dict[str, Any], max_route_index: int) -> RouteKinematicEnv:
    return RouteKinematicEnv(
        route=route,
        config=RouteEnvConfig(
            base_env_config=to_env_config(cfg),
            reset_config=_route_reset_config(cfg, max_route_index=max_route_index),
            reward_config=_route_reward_config(cfg),
            observation_config=RouteObservationConfig(**cfg.get("route", {}).get("observation", {})),
        ),
    )


def _predict(model: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=float).reshape(-1)


def _roll_one(env: RouteKinematicEnv, model: Any, *, initial_q: np.ndarray, goal_index: int, initial_dq=None, initial_prev_action=None) -> dict[str, Any]:
    options = {
        "route_index": int(goal_index),
        "start_route_index": 0,
        "initial_q": initial_q,
    }
    if initial_dq is not None:
        options["initial_dq"] = initial_dq
    if initial_prev_action is not None:
        options["initial_prev_action"] = initial_prev_action
    obs, info = env.reset(
        options={
            "route_index": int(goal_index),
            "start_route_index": 0,
            "policy_mode": "approach",
        }
    )
    # RouteKinematicEnv explicit reset uses route start by index; override base state for sequential chaining.
    if initial_q is not None:
        obs, info = env.base_env.reset(
            options={
                "initial_q": initial_q,
                "initial_dq": np.zeros_like(initial_q) if initial_dq is None else initial_dq,
                "initial_prev_action": np.zeros_like(initial_q) if initial_prev_action is None else initial_prev_action,
                "goal_q": env.route.waypoint(goal_index).q_goal,
                "policy_mode": "approach",
            }
        )
        env._route_index = int(goal_index)  # noqa: SLF001 - evaluator needs explicit sequential state
        env._start_route_index = 0  # noqa: SLF001
        env._ready_streak = 0  # noqa: SLF001
        env._prev_info = dict(info)  # noqa: SLF001
        obs = env._augment_obs(obs)  # noqa: SLF001
    terminated = truncated = False
    min_pos = float(info["position_error_norm"])
    min_ori = float(info["orientation_error_norm"])
    min_q = float(np.linalg.norm(env.route.waypoint(goal_index).q_goal - np.asarray(info["q"], dtype=float)))
    first_ready_step = None
    max_ready_streak = 0
    steps = 0
    while not (terminated or truncated):
        action = _predict(model, obs)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        min_pos = min(min_pos, float(info["position_error_norm"]))
        min_ori = min(min_ori, float(info["orientation_error_norm"]))
        min_q = min(min_q, float(info["route_q_error_norm"]))
        if bool(info["route_ready"]) and first_ready_step is None:
            first_ready_step = steps
        max_ready_streak = max(max_ready_streak, int(info["route_ready_streak"]))
    return {
        "route_index": int(goal_index),
        "success": bool(info["success"]),
        "route_ready_hit": bool(first_ready_step is not None),
        "route_ready_dwell": bool(max_ready_streak >= env.config.base_env_config.termination_config.success_dwell_steps),
        "first_ready_step": first_ready_step,
        "max_ready_streak": int(max_ready_streak),
        "steps": int(steps),
        "final_position_error": float(info["position_error_norm"]),
        "final_orientation_error": float(info["orientation_error_norm"]),
        "final_q_error": float(info["route_q_error_norm"]),
        "min_position_error": float(min_pos),
        "min_orientation_error": float(min_ori),
        "min_q_error": float(min_q),
        "final_action_magnitude": float(info["action_l2"]),
        "final_dq_norm": float(info["executed_delta_q_l2"]),
        "final_q": np.asarray(info["q"], dtype=float).tolist(),
        "final_dq": np.asarray(info["dq"], dtype=float).tolist(),
        "final_prev_action": np.asarray(env.base_env._prev_action, dtype=float).tolist(),  # noqa: SLF001
    }


def _summarize_rows(rows: list[dict[str, Any]], route: RouteDataset) -> dict[str, Any]:
    if not rows:
        return {"target_count": 0}
    first_failure = next((row for row in rows if not row["success"]), None)
    longest_prefix = 0
    for row in rows:
        if row["success"]:
            longest_prefix += 1
        else:
            break
    prefix_end = min(longest_prefix, len(route) - 1)
    prefix_distance = route.waypoint(prefix_end).route_progress_m - route.waypoint(0).route_progress_m
    return {
        "target_count": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "route_ready_hit_rate": float(np.mean([row["route_ready_hit"] for row in rows])),
        "route_ready_dwell_rate": float(np.mean([row["route_ready_dwell"] for row in rows])),
        "longest_success_prefix": int(longest_prefix),
        "cumulative_successful_route_distance_m": float(prefix_distance),
        "first_failure_index": None if first_failure is None else int(first_failure["route_index"]),
        "first_failure_reason": None if first_failure is None else _failure_reason(first_failure),
        "mean_final_position_error": float(np.mean([row["final_position_error"] for row in rows])),
        "mean_final_orientation_error": float(np.mean([row["final_orientation_error"] for row in rows])),
        "mean_final_q_error": float(np.mean([row["final_q_error"] for row in rows])),
        "max_final_position_error": float(np.max([row["final_position_error"] for row in rows])),
        "max_final_orientation_error": float(np.max([row["final_orientation_error"] for row in rows])),
    }


def _failure_reason(row: dict[str, Any]) -> str:
    if row["final_position_error"] > 0.010:
        return "position"
    if row["final_orientation_error"] > 0.150:
        return "orientation"
    if row.get("final_action_magnitude", 0.0) > 1.20 or row.get("final_dq_norm", 0.0) > 0.040:
        return "motion_action"
    if row["final_q_error"] > 0.500:
        return "q_error"
    if not row["route_ready_dwell"]:
        return "dwell_or_motion"
    return "unknown"


def _chunk_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = [(1, 40), (41, 80), (81, 120), (121, 180), (181, 260), (261, 360), (361, 483)]
    out: dict[str, Any] = {}
    for idx, (lo, hi) in enumerate(chunks):
        subset = [row for row in rows if lo <= row["route_index"] <= hi]
        if not subset:
            continue
        out[f"chunk_{idx}_{lo}_{hi}"] = {
            "target_count": len(subset),
            "success_rate": float(np.mean([row["success"] for row in subset])),
            "route_ready_hit_rate": float(np.mean([row["route_ready_hit"] for row in subset])),
            "mean_final_position_error": float(np.mean([row["final_position_error"] for row in subset])),
            "mean_final_orientation_error": float(np.mean([row["final_orientation_error"] for row in subset])),
            "mean_final_q_error": float(np.mean([row["final_q_error"] for row in subset])),
        }
    return out


def evaluate_sequential_route(
    *,
    checkpoint_path: Path,
    config_path: Path,
    route_path: Path,
    artifact_root: Path,
    end_index: int | None = None,
    start_index: int = 1,
) -> dict[str, Any]:
    cfg = _load_route_config(config_path)
    route = load_route_dataset(route_path)
    model = _load_sb3_model("ppo", checkpoint_path)
    env = _make_route_env(route=route, cfg=cfg, max_route_index=end_index or len(route) - 1)
    rows: list[dict[str, Any]] = []
    current_q = route.waypoint(max(start_index - 1, 0)).q_goal.copy()
    current_dq = np.zeros_like(current_q)
    current_prev_action = np.zeros_like(current_q)
    final_end = min(int(end_index or (len(route) - 1)), len(route) - 1)
    for idx in range(int(start_index), final_end + 1):
        row = _roll_one(
            env,
            model,
            initial_q=current_q,
            goal_index=idx,
            initial_dq=current_dq,
            initial_prev_action=current_prev_action,
        )
        rows.append({k: v for k, v in row.items() if k not in {"final_q", "final_dq", "final_prev_action"}})
        current_q = np.asarray(row["final_q"], dtype=float)
        current_dq = np.asarray(row["final_dq"], dtype=float)
        current_prev_action = np.asarray(row["final_prev_action"], dtype=float)
    summary = _summarize_rows(rows, route)
    summary.update(
        {
            "schema_version": "v5.route_curriculum.sequential_eval.v1",
            "mode": "sequential_actual_final_q_to_next_dense_q_goal",
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "route_path": str(route_path),
            "start_index": int(start_index),
            "end_index": int(final_end),
        }
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "route_eval_sequential_summary.json", summary)
    write_json(artifact_root / "route_chunk_metrics.json", _chunk_metrics(rows))
    with (artifact_root / "route_eval_sequential_steps.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    failure = next((row for row in rows if not row["success"]), None)
    write_json(
        artifact_root / "route_failure_report.json",
        {
            "first_failure_index": summary["first_failure_index"],
            "first_failure_reason": summary["first_failure_reason"],
            "first_failure": failure,
        },
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate route-curriculum checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--route-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = evaluate_sequential_route(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        route_path=Path(args.route_path),
        artifact_root=Path(args.artifact_root),
        start_index=args.start_index,
        end_index=args.end_index,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
