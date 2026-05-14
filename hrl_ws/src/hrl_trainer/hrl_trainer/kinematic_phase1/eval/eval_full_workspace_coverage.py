"""Random-start full workspace coverage evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..training.policy_config import approach_default_config_path, config_dir, deep_merge, load_yaml_file, repo_root, to_env_config, write_json
from ..workspace.adaptive_frontier_sampler import update_bucket_priorities
from ..workspace.start_target_pair_sampler import build_pair_sampler_summary, load_jsonl, write_pair_sampler_outputs
from ..workspace.workspace_start_state_map import generate_workspace_start_state_map, write_start_state_map
from ..workspace.workspace_target_map import generate_workspace_target_map, write_target_map
from .eval_approach_finisher import _finisher_ready
from .eval_deterministic import _load_sb3_model
from .eval_pipeline_ablation import _run_approach_with_handoff
from .eval_three_stage import _run_policy, _state_reset_options
from .eval_workspace_expansion import evaluate_workspace_expansion_checkpoint


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


def _mean(values: Sequence[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _reason(result: dict[str, Any], cfg, success: bool) -> str:
    if success:
        return "success"
    if float(result["final_position_error"]) > float(cfg.finisher_ready_pos_threshold_m):
        return "position"
    if float(result["final_orientation_error"]) > float(cfg.finisher_ready_ori_threshold_rad):
        return "orientation"
    if float(result["final_action_magnitude"]) > float(cfg.finisher_ready_action_threshold):
        return "motion_action"
    if float(result["final_dq_norm"]) > float(cfg.finisher_ready_dq_threshold):
        return "motion_dq"
    if not bool(result.get("dock_coarse_ready_dwell", False)):
        return "dwell"
    return "timeout_or_regression"


def _select_pairs(pairs: list[dict[str, Any]], *, mode: str, limit: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if mode == "known":
        pool = [p for p in pairs if int(p.get("target_stage_id") or 0) <= 8 and p.get("difficulty_class") in {"retention", "local", "medium"}]
    elif mode == "frontier":
        pool = [p for p in pairs if 8 <= int(p.get("target_stage_id") or 0) <= 11 and p.get("difficulty_class") in {"medium", "frontier", "stress"}]
    elif mode == "stress":
        pool = pairs
    else:
        raise ValueError(f"Unknown pair eval mode: {mode}")
    if not pool:
        pool = pairs
    if len(pool) <= limit:
        return list(pool)
    indices = rng.choice(len(pool), size=limit, replace=False)
    return [pool[int(i)] for i in indices]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        reasons[row["failure_reason"]] = reasons.get(row["failure_reason"], 0) + 1
        by_source.setdefault(str(row.get("start_source_type", "unknown")), []).append(row)
    return {
        "episode_count": len(rows),
        "success_rate": _mean([r["success"] for r in rows]),
        "ready_rate": _mean([r["finisher_ready_hit"] for r in rows]),
        "dwell_success_rate": _mean([r["finisher_ready_dwell"] for r in rows]),
        "mean_final_position_error": _mean([r["final_position_error"] for r in rows]),
        "mean_final_orientation_error": _mean([r["final_orientation_error"] for r in rows]),
        "mean_final_action_magnitude": _mean([r["final_action_magnitude"] for r in rows]),
        "mean_final_dq_norm": _mean([r["final_dq_norm"] for r in rows]),
        "average_start_target_joint_distance": _mean([r["joint_distance_l2"] for r in rows]),
        "average_start_target_ee_distance": _mean([r["ee_position_distance"] for r in rows]),
        "max_successful_joint_l2": max((r["joint_distance_l2"] for r in rows if r["success"]), default=0.0),
        "failure_reason_counts": reasons,
        "success_by_start_source": {
            source: {
                "episode_count": len(items),
                "success_rate": _mean([item["success"] for item in items]),
            }
            for source, items in by_source.items()
        },
    }


def _bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["target_bucket_id"]), []).append(row)
    metrics: dict[str, dict[str, Any]] = {}
    for bucket_id, items in grouped.items():
        metrics[bucket_id] = {
            "episode_count": len(items),
            "success_rate": _mean([item["success"] for item in items]),
            "failure_count": sum(1 for item in items if not item["success"]),
            "mean_final_position_error": _mean([item["final_position_error"] for item in items]),
            "mean_min_position_error": _mean([item["min_position_error"] for item in items]),
        }
    return metrics


def _run_pairs(
    *,
    pairs: list[dict[str, Any]],
    starts_by_id: dict[str, dict[str, Any]],
    targets_by_id: dict[str, dict[str, Any]],
    approach_model: Any,
    approach_env_cfg: Phase1EnvConfig,
    finisher_model: Any | None,
    finisher_env_cfg: Phase1EnvConfig | None,
    handoff_confirm_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        start = starts_by_id[pair["start_id"]]
        target = targets_by_id[pair["target_id"]]
        env = ArmKinematicEnv(config=approach_env_cfg)
        env.set_policy_mode("approach")
        reset_options = {
            "initial_q": start["q_start"],
            "initial_dq": start.get("dq_start", [0.0] * approach_env_cfg.n_joints),
            "initial_prev_action": start.get("prev_action", [0.0] * approach_env_cfg.n_joints),
            "goal_q": target["q_target"],
            "goal_pose6": [*target["ee_target_position"], *target["ee_target_orientation"]],
            "policy_mode": "approach",
        }
        approach_result, handoff_result = _run_approach_with_handoff(
            env=env,
            model=approach_model,
            reset_options=reset_options,
            ready_cfg=approach_env_cfg.reward_config,
            handoff_confirm_steps=handoff_confirm_steps,
        )
        final_ready = _finisher_ready(approach_result, cfg=approach_env_cfg.reward_config)
        handoff_result = approach_result if final_ready else handoff_result
        final_result = approach_result
        success = bool(approach_result["success"])
        if finisher_model is not None and finisher_env_cfg is not None and handoff_result is not None:
            finisher_env = ArmKinematicEnv(config=finisher_env_cfg)
            finisher_env.set_policy_mode("dock")
            final_result = _run_policy(
                env=finisher_env,
                model=finisher_model,
                reset_options=_state_reset_options(handoff_result, policy_mode="dock"),
            )
            success = bool(final_result["success"])
        row = {
            "episode_id": idx,
            "pair_id": pair["pair_id"],
            "start_id": pair["start_id"],
            "target_id": pair["target_id"],
            "start_source_type": pair.get("start_source_type"),
            "target_stage_id": pair.get("target_stage_id"),
            "target_bucket_id": pair.get("target_bucket_id"),
            "difficulty_class": pair.get("difficulty_class"),
            "joint_distance_l2": float(pair.get("joint_distance_l2", 0.0)),
            "ee_position_distance": float(pair.get("ee_position_distance", 0.0)),
            "success": bool(success),
            "finisher_ready_hit": bool(approach_result["dock_coarse_ready_hit"] or final_ready),
            "finisher_ready_dwell": bool(approach_result["dock_coarse_ready_dwell"] or final_ready),
            "failure_reason": _reason(approach_result, approach_env_cfg.reward_config, success),
            "final_position_error": float(final_result["final_position_error"]),
            "final_orientation_error": float(final_result["final_orientation_error"]),
            "approach_final_position_error": float(approach_result["final_position_error"]),
            "approach_final_orientation_error": float(approach_result["final_orientation_error"]),
            "min_position_error": float(approach_result["min_position_error"]),
            "min_orientation_error": float(approach_result["min_orientation_error"]),
            "final_action_magnitude": float(final_result["final_action_magnitude"]),
            "final_dq_norm": float(final_result["final_dq_norm"]),
        }
        rows.append(row)
    return rows


def evaluate_full_workspace_coverage(
    *,
    approach_checkpoint: Path,
    approach_config_path: Path,
    artifact_root: Path,
    finisher_checkpoint: Path | None = None,
    finisher_config_path: Path | None = None,
    seed: int = 940001,
    episodes_per_split: int = 96,
    stage_samples_per_stage: int = 96,
    random_target_samples: int = 384,
    random_start_samples: int = 384,
    pair_count: int = 2048,
    handoff_confirm_steps: int = 2,
    include_home_stage_eval: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    artifact_root.mkdir(parents=True, exist_ok=True)
    map_dir = artifact_root / "maps"
    target_samples, target_summary = generate_workspace_target_map(
        config_path=approach_config_path,
        seed=seed + 1,
        stage_samples_per_stage=stage_samples_per_stage,
        random_samples=random_target_samples,
    )
    start_samples, start_summary = generate_workspace_start_state_map(
        config_path=approach_config_path,
        seed=seed + 2,
        stage_samples_per_stage=max(stage_samples_per_stage // 2, 1),
        random_samples=random_start_samples,
    )
    write_target_map(target_samples, target_summary, map_dir)
    write_start_state_map(start_samples, start_summary, map_dir)
    pairs, pair_summary = build_pair_sampler_summary(
        start_map_path=map_dir / "start_state_map.jsonl",
        target_map_path=map_dir / "target_map.jsonl",
        seed=seed + 3,
        pair_count=pair_count,
    )
    write_pair_sampler_outputs(pairs, pair_summary, map_dir)

    cfg = deep_merge(load_yaml_file(approach_default_config_path()), _load_overlay_with_bases(approach_config_path))
    approach_env_cfg = to_env_config(cfg)
    finisher_env_cfg = to_env_config(load_yaml_file(finisher_config_path)) if finisher_config_path else None
    approach_model = _load_sb3_model("ppo", approach_checkpoint)
    finisher_model = _load_sb3_model("ppo", finisher_checkpoint) if finisher_checkpoint else None
    starts_by_id = {row["start_id"]: row for row in load_jsonl(map_dir / "start_state_map.jsonl")}
    targets_by_id = {row["target_id"]: row for row in load_jsonl(map_dir / "target_map.jsonl")}

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("known", "frontier", "stress"):
        selected_pairs = _select_pairs(pairs, mode=split, limit=episodes_per_split, rng=rng)
        rows = _run_pairs(
            pairs=selected_pairs,
            starts_by_id=starts_by_id,
            targets_by_id=targets_by_id,
            approach_model=approach_model,
            approach_env_cfg=approach_env_cfg,
            finisher_model=finisher_model,
            finisher_env_cfg=finisher_env_cfg,
            handoff_confirm_steps=handoff_confirm_steps,
        )
        split_rows[split] = rows
        write_json(artifact_root / f"{split}_random_start_eval_summary.json", {"summary": _summarize(rows), "episode_rows": rows})

    all_rows = [row for rows in split_rows.values() for row in rows]
    bucket_metrics = _bucket_metrics(all_rows)
    priorities = update_bucket_priorities(bucket_metrics)
    stable = sum(1 for data in bucket_metrics.values() if float(data["success_rate"]) >= 0.85)
    partial = sum(1 for data in bucket_metrics.values() if 0.35 <= float(data["success_rate"]) < 0.85)
    stress = sum(1 for data in bucket_metrics.values() if float(data["success_rate"]) < 0.35)
    coverage_summary = {
        "approach_checkpoint": str(approach_checkpoint),
        "finisher_checkpoint": str(finisher_checkpoint) if finisher_checkpoint else None,
        "target_map_summary": target_summary,
        "start_state_map_summary": start_summary,
        "pair_sampler_summary": pair_summary,
        "random_start_known_workspace": _summarize(split_rows["known"]),
        "random_start_frontier": _summarize(split_rows["frontier"]),
        "full_reachable_stress": _summarize(split_rows["stress"]),
        "covered_bucket_fraction": float((stable + partial) / max(len(bucket_metrics), 1)),
        "stable_bucket_fraction": float(stable / max(len(bucket_metrics), 1)),
        "partial_bucket_fraction": float(partial / max(len(bucket_metrics), 1)),
        "stress_bucket_fraction": float(stress / max(len(bucket_metrics), 1)),
        "covered_bucket_count": int(stable + partial),
        "total_eval_bucket_count": len(bucket_metrics),
        "top_sampling_priorities": [priority.__dict__ for priority in priorities[:30]],
    }
    write_json(artifact_root / "workspace_bucket_metrics.json", bucket_metrics)
    write_json(artifact_root / "full_workspace_coverage_summary.json", coverage_summary)
    write_json(
        artifact_root / "workspace_failure_report.json",
        {
            split: {
                "failure_reason_counts": _summarize(rows)["failure_reason_counts"],
                "worst_rows": sorted(rows, key=lambda row: row["final_position_error"] + 0.02 * row["final_orientation_error"], reverse=True)[:20],
            }
            for split, rows in split_rows.items()
        },
    )

    if include_home_stage_eval:
        home_eval = evaluate_workspace_expansion_checkpoint(
            approach_checkpoint=approach_checkpoint,
            approach_config_path=approach_config_path,
            artifact_root=artifact_root / "home_start_stage_eval",
            finisher_checkpoint=finisher_checkpoint,
            finisher_config_path=finisher_config_path,
            episodes=max(8, min(episodes_per_split // 4, 32)),
            seed=seed + 4,
            stage_indices=list(range(len(approach_env_cfg.curriculum_config.stages))),
        )
        coverage_summary["home_start_stage_eval_path"] = str(artifact_root / "home_start_stage_eval")
        coverage_summary["home_start_stage_metrics"] = home_eval["stage_metrics"]
        write_json(artifact_root / "full_workspace_coverage_summary.json", coverage_summary)
    return coverage_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mixed-start full workspace coverage.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--finisher-checkpoint")
    parser.add_argument("--finisher-config")
    parser.add_argument("--seed", type=int, default=940001)
    parser.add_argument("--episodes-per-split", type=int, default=96)
    parser.add_argument("--stage-samples-per-stage", type=int, default=96)
    parser.add_argument("--random-target-samples", type=int, default=384)
    parser.add_argument("--random-start-samples", type=int, default=384)
    parser.add_argument("--pair-count", type=int, default=2048)
    parser.add_argument("--skip-home-stage-eval", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = evaluate_full_workspace_coverage(
        approach_checkpoint=Path(args.approach_checkpoint),
        approach_config_path=Path(args.approach_config),
        artifact_root=Path(args.artifact_root),
        finisher_checkpoint=Path(args.finisher_checkpoint) if args.finisher_checkpoint else None,
        finisher_config_path=Path(args.finisher_config) if args.finisher_config else None,
        seed=args.seed,
        episodes_per_split=args.episodes_per_split,
        stage_samples_per_stage=args.stage_samples_per_stage,
        random_target_samples=args.random_target_samples,
        random_start_samples=args.random_start_samples,
        pair_count=args.pair_count,
        include_home_stage_eval=not args.skip_home_stage_eval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
