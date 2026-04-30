"""Label handoff candidate states by rolling out the dock policy from each state."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..eval.eval_deterministic import _load_sb3_model
from ..training.policy_config import deep_merge, dock_default_config_path, load_yaml_file, to_env_config, write_json
from .collect_handoff_dataset import handoff_default_config_path
from .handoff_dataset import read_jsonl, summarize_labeled_records, write_jsonl


def _load_config(explicit_path: str | None) -> dict:
    cfg = load_yaml_file(handoff_default_config_path())
    if explicit_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(explicit_path)))
    return cfg


def label_records_with_dock(
    *,
    dataset_path: Path,
    dock_checkpoint: Path,
    dock_config_path: Path,
    artifact_root: Path,
    config: dict,
    dock_algorithm: str = "td3",
) -> dict[str, object]:
    records = read_jsonl(dataset_path)
    dock_cfg = deep_merge(load_yaml_file(dock_default_config_path()), load_yaml_file(dock_config_path))
    env_config = to_env_config(dock_cfg)
    labeling_cfg = config.get("labeling", {})
    horizon = int(labeling_cfg.get("dock_rollout_horizon", env_config.termination_config.max_episode_steps))
    env_config = replace(env_config, termination_config=replace(env_config.termination_config, max_episode_steps=horizon))
    tolerance = float(labeling_cfg.get("regression_tolerance_m", 0.008))
    model = _load_sb3_model(dock_algorithm, dock_checkpoint)
    env = ArmKinematicEnv(config=env_config)
    labeled: list[dict] = []
    for idx, record in enumerate(records):
        obs, info = env.reset(
            options={
                "policy_mode": "dock",
                "initial_q": record["q"],
                "initial_dq": record.get("dq"),
                "initial_prev_action": record.get("prev_action"),
                "goal_q": record.get("goal_q"),
                "goal_pose6": record["goal_pose6"],
            }
        )
        terminated = truncated = False
        min_pos = float(info["position_error_norm"])
        min_ori = float(info["orientation_error_norm"])
        strict_pose_hit = (
            min_pos <= env.config.termination_config.success_pos_threshold_m
            and min_ori <= env.config.termination_config.success_ori_threshold_rad
        )
        action_norms: list[float] = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action, dtype=float)
            action_norms.append(float(np.linalg.norm(action_arr)))
            obs, _, terminated, truncated, info = env.step(action_arr)
            pos = float(info["position_error_norm"])
            ori = float(info["orientation_error_norm"])
            min_pos = min(min_pos, pos)
            min_ori = min(min_ori, ori)
            strict_pose_hit = strict_pose_hit or (
                pos <= env.config.termination_config.success_pos_threshold_m
                and ori <= env.config.termination_config.success_ori_threshold_rad
            )
        final_pos = float(info["position_error_norm"])
        final_ori = float(info["orientation_error_norm"])
        item = dict(record)
        item.update(
            {
                "label_index": idx,
                "dock_success_from_here": bool(info["success"]),
                "dock_strict_pose_hit_from_here": bool(strict_pose_hit),
                "dock_dwell_success_from_here": int(info["dwell_count"]) >= env.config.dwell_steps_target,
                "dock_final_position_error_from_here": final_pos,
                "dock_final_orientation_error_from_here": final_ori,
                "dock_min_position_error_from_here": min_pos,
                "dock_min_orientation_error_from_here": min_ori,
                "dock_regression_from_here": bool(final_pos > min_pos + tolerance),
                "dock_final_minus_min_position_error_from_here": final_pos - min_pos,
                "dock_action_l2_mean_from_here": float(np.mean(action_norms)) if action_norms else 0.0,
                "dock_checkpoint": str(dock_checkpoint),
            }
        )
        labeled.append(item)

    output_path = artifact_root / "handoff_labeled_dataset.jsonl"
    summary = summarize_labeled_records(labeled)
    summary.update(
        {
            "input_dataset_path": str(dataset_path),
            "labeled_dataset_path": str(output_path),
            "dock_checkpoint": str(dock_checkpoint),
            "dock_config_path": str(dock_config_path),
            "labeling_config": config,
        }
    )
    write_jsonl(output_path, labeled)
    write_json(artifact_root / "handoff_labeled_summary.json", summary)
    return {
        "labeled_dataset_path": str(output_path),
        "summary_path": str(artifact_root / "handoff_labeled_summary.json"),
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label Phase 1C handoff states with dock rollout outcomes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dock-checkpoint", required=True)
    parser.add_argument("--dock-config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--dock-algorithm", default="td3")
    return parser


def main() -> None:  # pragma: no cover
    args = build_arg_parser().parse_args()
    result = label_records_with_dock(
        dataset_path=Path(args.dataset),
        dock_checkpoint=Path(args.dock_checkpoint),
        dock_config_path=Path(args.dock_config),
        artifact_root=Path(args.artifact_root),
        config=_load_config(args.config),
        dock_algorithm=args.dock_algorithm,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
