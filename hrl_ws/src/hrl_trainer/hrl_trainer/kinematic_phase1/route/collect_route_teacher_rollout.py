"""Collect deterministic route teacher observations/actions for anchor training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.eval_deterministic import _load_sb3_model
from ..eval.eval_route_curriculum import _load_route_config, _make_route_env, _predict
from ..training.policy_config import write_json
from .route_dataset import load_route_dataset


def _jsonable_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float32).copy() for key, value in obs.items()}


def collect_teacher_rollout(
    *,
    checkpoint_path: Path,
    config_path: Path,
    route_path: Path,
    artifact_root: Path,
    start_index: int = 1,
    end_index: int = 120,
) -> dict[str, Any]:
    cfg = _load_route_config(config_path)
    route = load_route_dataset(route_path)
    model = _load_sb3_model("ppo", checkpoint_path)
    env = _make_route_env(route=route, cfg=cfg, max_route_index=end_index)

    obs_rows: dict[str, list[np.ndarray]] = {}
    action_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    current_q = route.waypoint(max(start_index - 1, 0)).q_goal.copy()
    current_dq = np.zeros_like(current_q)
    current_prev_action = np.zeros_like(current_q)
    successful_indices: list[int] = []
    failed_indices: list[int] = []

    for route_index in range(int(start_index), int(end_index) + 1):
        obs, info = env.reset(options={"route_index": int(route_index), "start_route_index": 0, "policy_mode": "approach"})
        obs, info = env.base_env.reset(
            options={
                "initial_q": current_q,
                "initial_dq": current_dq,
                "initial_prev_action": current_prev_action,
                "goal_q": route.waypoint(route_index).q_goal,
                "policy_mode": "approach",
            }
        )
        env._route_index = int(route_index)  # noqa: SLF001 - collection mirrors sequential evaluator
        env._start_route_index = 0  # noqa: SLF001
        env._ready_streak = 0  # noqa: SLF001
        env._prev_info = dict(info)  # noqa: SLF001
        obs = env._augment_obs(obs)  # noqa: SLF001

        terminated = truncated = False
        episode_start = len(action_rows)
        steps = 0
        while not (terminated or truncated):
            action = _predict(model, obs)
            obs_snapshot = _jsonable_obs(obs)
            for key, value in obs_snapshot.items():
                obs_rows.setdefault(key, []).append(value)
            action_rows.append(np.asarray(action, dtype=np.float32))
            meta_rows.append({"route_index": int(route_index), "step": int(steps)})
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

        if bool(info.get("success", False)):
            successful_indices.append(int(route_index))
            current_q = np.asarray(info["q"], dtype=float)
            current_dq = np.asarray(info["dq"], dtype=float)
            current_prev_action = np.asarray(env.base_env._prev_action, dtype=float)  # noqa: SLF001
        else:
            failed_indices.append(int(route_index))
            # Keep later sequential states honest, but do not use failed episode samples as teacher anchor.
            for key in list(obs_rows):
                del obs_rows[key][episode_start:]
            del action_rows[episode_start:]
            del meta_rows[episode_start:]
            current_q = np.asarray(info["q"], dtype=float)
            current_dq = np.asarray(info["dq"], dtype=float)
            current_prev_action = np.asarray(env.base_env._prev_action, dtype=float)  # noqa: SLF001
            break

    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_path = artifact_root / "teacher_route_anchor_dataset.npz"
    arrays: dict[str, np.ndarray] = {
        "actions": np.asarray(action_rows, dtype=np.float32),
        "route_index": np.asarray([row["route_index"] for row in meta_rows], dtype=np.int32),
        "step": np.asarray([row["step"] for row in meta_rows], dtype=np.int32),
    }
    for key, values in obs_rows.items():
        arrays[f"obs__{key}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(dataset_path, **arrays)

    summary = {
        "schema_version": "v5.route_teacher_anchor_dataset.v1",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "route_path": str(route_path),
        "dataset_path": str(dataset_path),
        "start_index": int(start_index),
        "requested_end_index": int(end_index),
        "successful_indices": successful_indices,
        "failed_indices": failed_indices,
        "sample_count": int(len(action_rows)),
        "obs_keys": sorted(obs_rows.keys()),
        "action_dim": int(arrays["actions"].shape[1]) if len(action_rows) else 0,
    }
    write_json(artifact_root / "teacher_route_anchor_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect route teacher anchor dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--route-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=120)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collect_teacher_rollout(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        route_path=Path(args.route_path),
        artifact_root=Path(args.artifact_root),
        start_index=args.start_index,
        end_index=args.end_index,
    )


if __name__ == "__main__":
    main()
