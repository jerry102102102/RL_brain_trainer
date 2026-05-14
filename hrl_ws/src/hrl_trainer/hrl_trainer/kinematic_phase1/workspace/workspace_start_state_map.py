"""Start-state map generation for random-start workspace training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..envs.curriculum import sample_stage_joint_target
from ..kinematics.fk_interface import compute_ee_pose6
from ..kinematics.joint_limits import joint_limit_margin, sample_joint_configuration
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
class WorkspaceStartStateSample:
    start_id: str
    q_start: list[float]
    dq_start: list[float]
    prev_action: list[float]
    ee_position: list[float]
    ee_orientation: list[float]
    source_type: str
    source_stage: int | None
    source_rollout_id: str | None
    stability_score: float
    joint_limit_margin_min: float
    bucket_id: str


def _bucket_id(pose6: np.ndarray, q: np.ndarray, margin_min: float) -> str:
    x = int(np.clip(np.floor((pose6[0] + 1.0) / 2.0 * 8), 0, 7))
    y = int(np.clip(np.floor((pose6[1] + 1.0) / 2.0 * 8), 0, 7))
    z = int(np.clip(np.floor((pose6[2]) / 2.0 * 6), 0, 5))
    q_bucket = int(np.clip(np.floor(np.linalg.norm(q) / 4.5 * 6), 0, 5))
    m_bucket = int(np.clip(np.floor(margin_min * 5), 0, 4))
    return f"x{x}_y{y}_z{z}_q{q_bucket}_m{m_bucket}"


def _stability_score(margin_min: float, dq: np.ndarray, prev_action: np.ndarray) -> float:
    motion = min(float(np.linalg.norm(dq)) + float(np.linalg.norm(prev_action)), 1.0)
    return float(0.7 * np.clip(margin_min, 0.0, 1.0) + 0.3 * (1.0 - motion))


def generate_workspace_start_state_map(
    *,
    config_path: Path,
    seed: int,
    stage_samples_per_stage: int,
    random_samples: int,
    stage_indices: Sequence[int] | None = None,
    dq_noise: float = 0.001,
    prev_action_noise: float = 0.03,
) -> tuple[list[WorkspaceStartStateSample], dict[str, Any]]:
    cfg = deep_merge(load_yaml_file(approach_default_config_path()), _load_overlay_with_bases(config_path))
    env_cfg = to_env_config(cfg)
    rng = np.random.default_rng(seed)
    stages = env_cfg.curriculum_config.stages
    selected = list(stage_indices) if stage_indices is not None else list(range(len(stages)))
    selected = [int(np.clip(i, 0, len(stages) - 1)) for i in selected]
    raw: list[tuple[np.ndarray, str, int | None, str | None]] = [(np.zeros(env_cfg.n_joints, dtype=float), "home", 0, None)]

    for stage_id in selected:
        stage = stages[stage_id]
        for sample_idx in range(max(stage_samples_per_stage, 0)):
            if rng.random() < 0.65:
                q = sample_stage_joint_target(rng, stage.goal_q, stage.goal_noise, env_cfg.joint_specs)
                source = "successful_rollout"
            else:
                q = sample_stage_joint_target(rng, stage.start_q, stage.start_noise, env_cfg.joint_specs)
                source = "near_target" if stage_id >= 6 else "successful_rollout"
            raw.append((q, source, stage_id, f"stage{stage_id:02d}_synthetic_{sample_idx:04d}"))

    for sample_idx in range(max(random_samples, 0)):
        raw.append((sample_joint_configuration(rng, env_cfg.joint_specs, margin_fraction=0.10), "random_valid_q", None, f"random_{sample_idx:04d}"))

    samples: list[WorkspaceStartStateSample] = []
    for idx, (q, source, stage_id, rollout_id) in enumerate(raw):
        dq = rng.uniform(-dq_noise, dq_noise, size=env_cfg.n_joints)
        prev_action = rng.uniform(-prev_action_noise, prev_action_noise, size=env_cfg.n_joints)
        if source == "home":
            dq = np.zeros(env_cfg.n_joints, dtype=float)
            prev_action = np.zeros(env_cfg.n_joints, dtype=float)
        pose6 = compute_ee_pose6(q)
        margin_min = float(np.min(joint_limit_margin(q, env_cfg.joint_specs)))
        samples.append(
            WorkspaceStartStateSample(
                start_id=f"start_{idx:06d}",
                q_start=q.astype(float).tolist(),
                dq_start=dq.astype(float).tolist(),
                prev_action=prev_action.astype(float).tolist(),
                ee_position=pose6[:3].astype(float).tolist(),
                ee_orientation=pose6[3:].astype(float).tolist(),
                source_type=source,
                source_stage=stage_id,
                source_rollout_id=rollout_id,
                stability_score=_stability_score(margin_min, dq, prev_action),
                joint_limit_margin_min=margin_min,
                bucket_id=_bucket_id(pose6, q, margin_min),
            )
        )

    source_counts: dict[str, int] = {}
    for sample in samples:
        source_counts[sample.source_type] = source_counts.get(sample.source_type, 0) + 1
    pos_stack = np.vstack([np.asarray(s.ee_position, dtype=float) for s in samples]) if samples else np.zeros((0, 3))
    summary = {
        "seed": int(seed),
        "total_start_count": len(samples),
        "source_counts": source_counts,
        "bucket_count": len({s.bucket_id for s in samples}),
        "xyz_span": (pos_stack.max(axis=0) - pos_stack.min(axis=0)).tolist() if len(pos_stack) else [0.0, 0.0, 0.0],
        "joint_limit_margin_min": float(min((s.joint_limit_margin_min for s in samples), default=0.0)),
        "joint_limit_margin_mean": float(np.mean([s.joint_limit_margin_min for s in samples])) if samples else 0.0,
        "random_start_note": "Start states intentionally include non-home q states; this is the core distinction from prior home-start stage sweeps.",
    }
    return samples, summary


def write_start_state_map(samples: Sequence[WorkspaceStartStateSample], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "start_state_map.jsonl").write_text("\n".join(json.dumps(asdict(s)) for s in samples) + ("\n" if samples else ""))
    (output_dir / "start_state_map_summary.json").write_text(json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a mixed workspace start-state map.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=920001)
    parser.add_argument("--stage-samples-per-stage", type=int, default=96)
    parser.add_argument("--random-samples", type=int, default=384)
    parser.add_argument("--stages", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stages = [int(x) for x in args.stages.split(",") if x.strip()] if args.stages else None
    samples, summary = generate_workspace_start_state_map(
        config_path=Path(args.config),
        seed=args.seed,
        stage_samples_per_stage=args.stage_samples_per_stage,
        random_samples=args.random_samples,
        stage_indices=stages,
    )
    write_start_state_map(samples, summary, Path(args.output_dir))
    print(json.dumps({"output_dir": str(Path(args.output_dir)), **summary}, indent=2))


if __name__ == "__main__":
    main()
