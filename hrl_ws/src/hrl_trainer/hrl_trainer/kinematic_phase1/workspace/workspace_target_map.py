"""Reachable target-map generation for full workspace coverage training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..envs.curriculum import sample_stage_joint_target
from ..kinematics.fk_interface import compute_ee_pose6
from ..kinematics.joint_limits import JointSpec, joint_limit_margin, sample_joint_configuration
from ..training.policy_config import approach_default_config_path, config_dir, deep_merge, load_yaml_file, repo_root, to_env_config


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


@dataclass(frozen=True)
class WorkspaceTargetSample:
    target_id: str
    q_target: list[float]
    ee_target_position: list[float]
    ee_target_orientation: list[float]
    stage_id: int | None
    source_type: str
    bucket_id: str
    xyz_bucket: list[int]
    orientation_bucket: int
    joint_l2_bucket: int
    joint_limit_margin_min: float
    reachability_flag: bool
    difficulty_score: float
    previous_eval_success_rate: float | None = None
    previous_failure_reason_counts: dict[str, int] | None = None


def _bucketize(values: Sequence[float], *, lower: Sequence[float], upper: Sequence[float], bins: int) -> list[int]:
    arr = np.asarray(values, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    scaled = (arr - lo) / np.maximum(hi - lo, 1e-9)
    return np.clip(np.floor(scaled * bins), 0, bins - 1).astype(int).tolist()


def _q_l2_bucket(q: np.ndarray, bins: int) -> int:
    value = float(np.linalg.norm(q))
    return int(np.clip(np.floor(value / 4.5 * bins), 0, bins - 1))


def _ori_bucket(pose6: np.ndarray, bins: int) -> int:
    value = float(np.linalg.norm(pose6[3:]))
    return int(np.clip(np.floor(value / np.pi * bins), 0, bins - 1))


def _difficulty(q: np.ndarray, pose6: np.ndarray, margin_min: float) -> float:
    q_term = min(float(np.linalg.norm(q)) / 4.5, 1.0)
    ori_term = min(float(np.linalg.norm(pose6[3:])) / np.pi, 1.0)
    margin_term = 1.0 - float(np.clip(margin_min, 0.0, 1.0))
    return float(0.45 * q_term + 0.35 * ori_term + 0.20 * margin_term)


def generate_workspace_target_map(
    *,
    config_path: Path,
    seed: int,
    stage_samples_per_stage: int,
    random_samples: int,
    stage_indices: Sequence[int] | None = None,
    xyz_bins: int = 8,
    ori_bins: int = 6,
    q_l2_bins: int = 6,
) -> tuple[list[WorkspaceTargetSample], dict[str, Any]]:
    cfg = deep_merge(load_yaml_file(approach_default_config_path()), _load_overlay_with_bases(config_path))
    env_cfg = to_env_config(cfg)
    rng = np.random.default_rng(seed)
    stages = env_cfg.curriculum_config.stages
    selected = list(stage_indices) if stage_indices is not None else list(range(len(stages)))
    selected = [int(np.clip(i, 0, len(stages) - 1)) for i in selected]
    samples: list[WorkspaceTargetSample] = []
    poses: list[np.ndarray] = []

    raw_q: list[tuple[np.ndarray, int | None, str]] = []
    for stage_id in selected:
        stage = stages[stage_id]
        for _ in range(max(stage_samples_per_stage, 0)):
            raw_q.append((sample_stage_joint_target(rng, stage.goal_q, stage.goal_noise, env_cfg.joint_specs), stage_id, "stage_distribution"))
    for _ in range(max(random_samples, 0)):
        raw_q.append((sample_joint_configuration(rng, env_cfg.joint_specs, margin_fraction=0.08), None, "random_valid_q"))

    for q, _, _ in raw_q:
        poses.append(compute_ee_pose6(q))
    if poses:
        pos_stack = np.vstack([p[:3] for p in poses])
        xyz_lower = pos_stack.min(axis=0) - 1e-6
        xyz_upper = pos_stack.max(axis=0) + 1e-6
    else:
        xyz_lower = np.asarray([-1.0, -1.0, 0.0])
        xyz_upper = np.asarray([1.0, 1.0, 2.0])

    for idx, (q, stage_id, source_type) in enumerate(raw_q):
        pose6 = compute_ee_pose6(q)
        margin_min = float(np.min(joint_limit_margin(q, env_cfg.joint_specs)))
        xyz_bucket = _bucketize(pose6[:3], lower=xyz_lower, upper=xyz_upper, bins=xyz_bins)
        ori_bucket = _ori_bucket(pose6, ori_bins)
        q_bucket = _q_l2_bucket(q, q_l2_bins)
        bucket_id = f"x{xyz_bucket[0]}_y{xyz_bucket[1]}_z{xyz_bucket[2]}_o{ori_bucket}_q{q_bucket}"
        samples.append(
            WorkspaceTargetSample(
                target_id=f"target_{idx:06d}",
                q_target=q.astype(float).tolist(),
                ee_target_position=pose6[:3].astype(float).tolist(),
                ee_target_orientation=pose6[3:].astype(float).tolist(),
                stage_id=stage_id,
                source_type=source_type,
                bucket_id=bucket_id,
                xyz_bucket=xyz_bucket,
                orientation_bucket=ori_bucket,
                joint_l2_bucket=q_bucket,
                joint_limit_margin_min=margin_min,
                reachability_flag=bool(margin_min > 0.0),
                difficulty_score=_difficulty(q, pose6, margin_min),
            )
        )

    bucket_count = len({s.bucket_id for s in samples})
    q_stack = np.vstack([np.asarray(s.q_target, dtype=float) for s in samples]) if samples else np.zeros((0, len(env_cfg.joint_specs)))
    pos_stack = np.vstack([np.asarray(s.ee_target_position, dtype=float) for s in samples]) if samples else np.zeros((0, 3))
    summary = {
        "seed": int(seed),
        "total_target_count": len(samples),
        "valid_target_count": sum(1 for s in samples if s.reachability_flag),
        "rejected_target_count": 0,
        "stage_indices": selected,
        "xyz_span": (pos_stack.max(axis=0) - pos_stack.min(axis=0)).tolist() if len(pos_stack) else [0.0, 0.0, 0.0],
        "xyz_min": pos_stack.min(axis=0).tolist() if len(pos_stack) else [0.0, 0.0, 0.0],
        "xyz_max": pos_stack.max(axis=0).tolist() if len(pos_stack) else [0.0, 0.0, 0.0],
        "q_l2_range": [float(np.min(np.linalg.norm(q_stack, axis=1))), float(np.max(np.linalg.norm(q_stack, axis=1)))] if len(q_stack) else [0.0, 0.0],
        "joint_limit_margin_min": float(min((s.joint_limit_margin_min for s in samples), default=0.0)),
        "joint_limit_margin_mean": float(np.mean([s.joint_limit_margin_min for s in samples])) if samples else 0.0,
        "bucket_count": bucket_count,
        "stage_is_workspace_note": "Stage IDs are difficulty shells, not the full continuous workspace.",
    }
    return samples, summary


def write_target_map(samples: Sequence[WorkspaceTargetSample], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "target_map.jsonl").write_text("\n".join(json.dumps(asdict(s)) for s in samples) + ("\n" if samples else ""))
    (output_dir / "target_map_summary.json").write_text(json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a sampled reachable workspace target map.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=910001)
    parser.add_argument("--stage-samples-per-stage", type=int, default=128)
    parser.add_argument("--random-samples", type=int, default=512)
    parser.add_argument("--stages", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stages = [int(x) for x in args.stages.split(",") if x.strip()] if args.stages else None
    samples, summary = generate_workspace_target_map(
        config_path=Path(args.config),
        seed=args.seed,
        stage_samples_per_stage=args.stage_samples_per_stage,
        random_samples=args.random_samples,
        stage_indices=stages,
    )
    write_target_map(samples, summary, Path(args.output_dir))
    print(json.dumps({"output_dir": str(Path(args.output_dir)), **summary}, indent=2))


if __name__ == "__main__":
    main()
