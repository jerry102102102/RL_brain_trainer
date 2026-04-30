"""Build a controlled perturbation map of the Dock policy acceptance basin."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..envs.reset_samplers import sample_dock_reset
from ..eval.eval_deterministic import _load_sb3_model
from ..kinematics.fk_interface import compute_ee_pose6
from ..training.policy_config import deep_merge, dock_default_config_path, load_yaml_file, to_env_config, write_json
from .dock_acceptance_analysis import bucket_label, save_acceptance_map, summarize_acceptance_records, write_acceptance_heatmap


def acceptance_default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "dock_acceptance_default.yaml"


def _load_config(explicit_path: str | None) -> dict:
    cfg = load_yaml_file(acceptance_default_config_path())
    if explicit_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(explicit_path)))
    return cfg


def _random_unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    vec = rng.normal(size=dim)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


def _sample_radius(rng: np.random.Generator, bucket: list[float]) -> float:
    lo, hi = float(bucket[0]), float(bucket[1])
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def _rollout_dock_from_state(
    *,
    env: ArmKinematicEnv,
    model,
    initial_q: np.ndarray,
    goal_q: np.ndarray,
    goal_pose6: np.ndarray,
    initial_dq: np.ndarray,
    initial_prev_action: np.ndarray,
    regression_tolerance_m: float,
) -> dict[str, object]:
    obs, info = env.reset(
        options={
            "policy_mode": "dock",
            "initial_q": initial_q,
            "initial_dq": initial_dq,
            "initial_prev_action": initial_prev_action,
            "goal_q": goal_q,
            "goal_pose6": goal_pose6,
        }
    )
    terminated = truncated = False
    min_pos = float(info["position_error_norm"])
    min_ori = float(info["orientation_error_norm"])
    strict_pose_hit = (
        min_pos <= env.config.termination_config.success_pos_threshold_m
        and min_ori <= env.config.termination_config.success_ori_threshold_rad
    )
    actions: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action_arr = np.asarray(action, dtype=float)
        actions.append(float(np.linalg.norm(action_arr)))
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
    return {
        "dock_success_from_here": bool(info["success"]),
        "dock_strict_pose_hit": bool(strict_pose_hit),
        "dock_dwell_success": int(info["dwell_count"]) >= env.config.dwell_steps_target,
        "dock_final_position_error": final_pos,
        "dock_final_orientation_error": float(info["orientation_error_norm"]),
        "dock_min_position_error": min_pos,
        "dock_min_orientation_error": min_ori,
        "dock_regression": bool(final_pos > min_pos + regression_tolerance_m),
        "dock_final_minus_min_position_error": final_pos - min_pos,
        "rollout_length": int(info.get("step_count", len(actions))),
        "dock_action_l2_mean": float(np.mean(actions)) if actions else 0.0,
    }


def build_dock_acceptance_map(
    *,
    dock_checkpoint: Path,
    dock_config_path: Path,
    artifact_root: Path,
    config: dict,
    dock_algorithm: str = "td3",
) -> dict[str, object]:
    dock_cfg = deep_merge(load_yaml_file(dock_default_config_path()), load_yaml_file(dock_config_path))
    env_config = to_env_config(dock_cfg)
    labeling_cfg = config.get("labeling", {})
    horizon = int(labeling_cfg.get("dock_rollout_horizon", env_config.termination_config.max_episode_steps))
    env_config = replace(env_config, termination_config=replace(env_config.termination_config, max_episode_steps=horizon))
    acceptance_cfg = config.get("acceptance", {})
    rng = np.random.default_rng(int(acceptance_cfg.get("seed", 700777)))
    model = _load_sb3_model(dock_algorithm, dock_checkpoint)
    env = ArmKinematicEnv(config=env_config)
    pos_buckets = list(acceptance_cfg.get("position_buckets_m", [[0.0, 0.005], [0.005, 0.010], [0.010, 0.020]]))
    ori_buckets = list(acceptance_cfg.get("orientation_buckets_rad", [[0.0, 0.05], [0.05, 0.2], [0.2, 0.5]]))
    dq_norm_values = [float(v) for v in acceptance_cfg.get("dq_norm_values", [0.0])]
    prev_action_norm_values = [float(v) for v in acceptance_cfg.get("prev_action_norm_values", [0.0])]
    samples_per_bucket = int(acceptance_cfg.get("samples_per_bucket", 4))
    base_state_count = int(acceptance_cfg.get("base_state_count", 20))

    base_states = []
    for base_id in range(base_state_count):
        sample = sample_dock_reset(
            rng=rng,
            joint_specs=env_config.joint_specs,
            dock_reset_config=env_config.dock_reset_config,
            curriculum_config=env_config.curriculum_config,
            stage_index=0,
        )
        base_q = np.asarray(sample.goal_q, dtype=float)
        base_states.append((base_id, base_q, compute_ee_pose6(base_q)))

    records: list[dict[str, object]] = []
    sample_id = 0
    for base_id, base_q, base_pose6 in base_states:
        for pos_bucket in pos_buckets:
            for ori_bucket in ori_buckets:
                for dq_norm in dq_norm_values:
                    for prev_action_norm in prev_action_norm_values:
                        for _ in range(samples_per_bucket):
                            pos_radius = _sample_radius(rng, pos_bucket)
                            ori_radius = _sample_radius(rng, ori_bucket)
                            pos_offset = _random_unit_vector(rng, 3) * pos_radius
                            ori_offset = _random_unit_vector(rng, 3) * ori_radius
                            goal_pose6 = np.asarray(base_pose6, dtype=float).copy()
                            goal_pose6[:3] = goal_pose6[:3] + pos_offset
                            goal_pose6[3:] = goal_pose6[3:] + ori_offset
                            initial_dq = _random_unit_vector(rng, 7) * dq_norm if dq_norm > 0.0 else np.zeros(7)
                            initial_prev_action = (
                                _random_unit_vector(rng, 7) * prev_action_norm if prev_action_norm > 0.0 else np.zeros(7)
                            )
                            rollout = _rollout_dock_from_state(
                                env=env,
                                model=model,
                                initial_q=base_q,
                                goal_q=base_q,
                                goal_pose6=goal_pose6,
                                initial_dq=initial_dq,
                                initial_prev_action=initial_prev_action,
                                regression_tolerance_m=float(labeling_cfg.get("regression_tolerance_m", 0.008)),
                            )
                            records.append(
                                {
                                    "sample_id": sample_id,
                                    "base_state_id": base_id,
                                    "perturbed_position_error": pos_radius,
                                    "perturbed_orientation_error": ori_radius,
                                    "perturbed_dq_norm": dq_norm,
                                    "perturbed_prev_action_norm": prev_action_norm,
                                    "position_bucket": bucket_label(float(pos_bucket[0]), float(pos_bucket[1]), "m"),
                                    "orientation_bucket": bucket_label(float(ori_bucket[0]), float(ori_bucket[1]), "rad"),
                                    "dq_bucket": f"{dq_norm:.3f}",
                                    "prev_action_bucket": f"{prev_action_norm:.3f}",
                                    "initial_q": base_q.tolist(),
                                    "goal_pose6": goal_pose6.tolist(),
                                    "source_dock_checkpoint": str(dock_checkpoint),
                                    **rollout,
                                }
                            )
                            sample_id += 1

    output_path = artifact_root / "dock_acceptance_map.jsonl"
    save_acceptance_map(output_path, records)
    summary = summarize_acceptance_records(records)
    heatmap_path = write_acceptance_heatmap(records, artifact_root / "plots" / "dock_acceptance_heatmap.png")
    summary.update(
        {
            "acceptance_map_path": str(output_path),
            "dock_checkpoint": str(dock_checkpoint),
            "dock_config_path": str(dock_config_path),
            "acceptance_config": config,
            "heatmap_path": heatmap_path,
        }
    )
    write_json(artifact_root / "dock_acceptance_map_summary.json", summary)
    return {"acceptance_map_path": str(output_path), "summary_path": str(artifact_root / "dock_acceptance_map_summary.json"), "summary": summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Phase 1C Dock acceptance map.")
    parser.add_argument("--dock-checkpoint", required=True)
    parser.add_argument("--dock-config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--dock-algorithm", default="td3")
    return parser


def main() -> None:  # pragma: no cover
    args = build_arg_parser().parse_args()
    result = build_dock_acceptance_map(
        dock_checkpoint=Path(args.dock_checkpoint),
        dock_config_path=Path(args.dock_config),
        artifact_root=Path(args.artifact_root),
        config=_load_config(args.config),
        dock_algorithm=args.dock_algorithm,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
