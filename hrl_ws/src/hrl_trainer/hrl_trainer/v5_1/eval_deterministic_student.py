"""Evaluate a deterministic student policy against fixed teacher baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .curriculum import resolve_stages
from .deterministic_student import (
    DEFAULT_TEACHER_COMPARE_RUNS,
    load_student_checkpoint,
    parse_csv_list,
)
from .pipeline_e2e import (
    _CONTROLLED_ACTION_DIM,
    _build_fixed_eval_suite,
    _controlled_joint_indices,
    _parse_gap_eval_scales,
    _reward_config_for_profile,
    _resolve_ee_target_from_external_task,
    _run_gap_diagnosis_gz,
    _run_post_training_eval_gz,
)
from .runtime_ros2 import RuntimeROS2Adapter


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_artifact_path(root: Path, spec: str | None) -> Path | None:
    raw = str(spec or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (root / path)


def _load_teacher_reference(repo_root: Path, source_root: Path, run_id: str) -> dict[str, Any]:
    run_root = source_root / str(run_id)
    pipeline_summary = _read_json(run_root / "pipeline_summary.json")
    fixed_eval_suite = dict(pipeline_summary.get("fixed_eval_suite", {}) or {})
    deterministic_summary_path = _resolve_artifact_path(repo_root, (pipeline_summary.get("artifacts", {}) or {}).get("post_train_eval_summary"))
    gap_summary_path = _resolve_artifact_path(repo_root, (pipeline_summary.get("artifacts", {}) or {}).get("gap_eval_summary"))
    deterministic_summary = _read_json(deterministic_summary_path) if deterministic_summary_path and deterministic_summary_path.exists() else {}
    gap_summary = _read_json(gap_summary_path) if gap_summary_path and gap_summary_path.exists() else {}
    return {
        "run_id": str(run_id),
        "run_root": str(run_root),
        "suite_id": str(fixed_eval_suite.get("suite_id", "")),
        "best_checkpoint_episode": pipeline_summary.get("best_checkpoint_episode"),
        "best_checkpoint_path": pipeline_summary.get("best_checkpoint_path"),
        "deterministic_metrics": dict(deterministic_summary.get("metrics", {}) or {}),
        "gap_metrics": dict(gap_summary.get("gap_metrics", {}) or {}),
    }


def _best_teacher_metrics(teachers: list[dict[str, Any]]) -> dict[str, float]:
    if not teachers:
        return {
            "true_outer_hit_rate": 0.0,
            "true_inner_hit_rate": 0.0,
            "true_dwell_hit_rate": 0.0,
            "true_basin_hit_rate": 0.0,
            "mean_final_dpos": 0.0,
            "regression_rate": 1.0,
            "success_rate": 0.0,
        }
    return {
        "true_outer_hit_rate": max(float(t["deterministic_metrics"].get("true_outer_hit_rate", 0.0)) for t in teachers),
        "true_inner_hit_rate": max(float(t["deterministic_metrics"].get("true_inner_hit_rate", 0.0)) for t in teachers),
        "true_dwell_hit_rate": max(float(t["deterministic_metrics"].get("true_dwell_hit_rate", 0.0)) for t in teachers),
        "true_basin_hit_rate": max(float(t["deterministic_metrics"].get("true_basin_hit_rate", 0.0)) for t in teachers),
        "mean_final_dpos": min(float(t["deterministic_metrics"].get("mean_final_dpos", 1.0e9)) for t in teachers),
        "regression_rate": min(float(t["deterministic_metrics"].get("regression_rate", 1.0)) for t in teachers),
        "success_rate": max(float(t["deterministic_metrics"].get("success_rate", 0.0)) for t in teachers),
    }


def evaluate_deterministic_student(
    *,
    student_checkpoint: Path,
    artifact_root: Path,
    runtime_mode: str,
    stage_profile: str,
    target_mode: str,
    source_root: Path,
    compare_runs: list[str],
    eval_suite_run: str | None,
    runtime_joint_names: list[str],
    trajectory_topic: str,
    joint_state_topic: str,
    reward_profile: str,
    external_task_prop: str,
    external_task_src_idx: int,
    external_task_dst_idx: int,
    external_task_waypoint_index: int,
    near_home_profile: str,
    near_home_pos_offset_min_m: float,
    near_home_pos_offset_max_m: float,
    near_home_ori_offset_min_deg: float,
    near_home_ori_offset_max_deg: float,
    gz_world_name: str,
    gz_target_entity_name: str,
    gz_target_world_offset_xyz: tuple[float, float, float],
    post_train_eval_episodes: int,
    post_train_eval_steps_per_episode: int,
    gap_eval_scales: str,
    gap_eval_episodes: int,
    gap_eval_steps_per_episode: int,
    ee_pos_success_threshold: float,
    ee_ori_success_threshold: float,
    reset_near_home_eps: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if str(runtime_mode) != "gz":
        raise ValueError("deterministic student evaluation currently supports runtime_mode=gz only")

    repo_root = Path.cwd()
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    policy, payload = load_student_checkpoint(student_checkpoint, device=str(device))
    payload_meta = dict(payload.get("metadata", {}) or {})
    student_run_id = str(payload.get("run_id", Path(student_checkpoint).stem))

    compare_runs = list(compare_runs)
    if not compare_runs:
        compare_runs = list(DEFAULT_TEACHER_COMPARE_RUNS)
    teacher_refs = [_load_teacher_reference(repo_root, Path(source_root), run_id) for run_id in compare_runs]

    suite_run = str(eval_suite_run or compare_runs[0])
    suite_summary = _read_json(Path(source_root) / suite_run / "pipeline_summary.json")
    fixed_eval_suite = dict(suite_summary.get("fixed_eval_suite", {}) or {})
    if not fixed_eval_suite:
        stages = resolve_stages(stage_profile)
        stage = stages[0]
        external_ee_target, external_ee_target_source = _resolve_ee_target_from_external_task(
            prop=external_task_prop,
            src_idx=external_task_src_idx,
            dst_idx=external_task_dst_idx,
            waypoint_index=external_task_waypoint_index,
        )
        fixed_eval_suite = _build_fixed_eval_suite(
            suite_size=max(1, int(max(post_train_eval_episodes, gap_eval_episodes or post_train_eval_episodes))),
            suite_seed=int(seed) + 700_001,
            target_mode=target_mode,
            action_stage_name=str(stage.name),
            target_curriculum_stage_name="TC0",
            near_home_profile=str(near_home_profile),
            near_home_pos_offset_min_m=float(near_home_pos_offset_min_m),
            near_home_pos_offset_max_m=float(near_home_pos_offset_max_m),
            near_home_ori_offset_min_deg=float(near_home_ori_offset_min_deg),
            near_home_ori_offset_max_deg=float(near_home_ori_offset_max_deg),
            external_ee_target=external_ee_target,
            external_ee_target_source=external_ee_target_source,
        )

    external_ee_target, external_ee_target_source = _resolve_ee_target_from_external_task(
        prop=external_task_prop,
        src_idx=external_task_src_idx,
        dst_idx=external_task_dst_idx,
        waypoint_index=external_task_waypoint_index,
    )

    stages = resolve_stages(stage_profile)
    stage = stages[0]
    runtime = RuntimeROS2Adapter.from_ros2(
        joint_names=list(runtime_joint_names),
        trajectory_topic=str(trajectory_topic),
        joint_state_topic=str(joint_state_topic),
        gz_visualize_target=True,
        gz_world_name=str(gz_world_name),
        gz_target_entity_name=str(gz_target_entity_name),
        gz_target_world_offset_xyz=tuple(float(v) for v in gz_target_world_offset_xyz),
    )
    runtime_controlled_indices = _controlled_joint_indices(runtime_joint_names)
    if len(runtime_controlled_indices) != int(_CONTROLLED_ACTION_DIM):
        raise ValueError(f"expected {_CONTROLLED_ACTION_DIM} controlled joints, got {len(runtime_controlled_indices)}")

    reward_profile_resolved = str(payload_meta.get("reward_profile", reward_profile))
    action_scale = float((payload.get("student_config", {}) or {}).get("action_scale", policy.cfg.action_scale))
    reward_config = _reward_config_for_profile(reward_profile_resolved, action_scale=float(action_scale))

    run_id = f"{student_run_id}_eval"
    post_eval_root = artifact_root / "eval"
    gap_eval_specs = _parse_gap_eval_scales(gap_eval_scales)
    if not gap_eval_specs:
        gap_eval_specs = _parse_gap_eval_scales("det,0.10,0.30,0.60")

    try:
        eval_outputs = _run_post_training_eval_gz(
            run_id=run_id,
            artifact_root=artifact_root,
            eval_root=post_eval_root,
            agent=policy,
            runtime=runtime,
            stage=stage,
            target_mode=target_mode,
            near_home_profile=near_home_profile,
            near_home_pos_offset_min_m=float(near_home_pos_offset_min_m),
            near_home_pos_offset_max_m=float(near_home_pos_offset_max_m),
            near_home_ori_offset_min_deg=float(near_home_ori_offset_min_deg),
            near_home_ori_offset_max_deg=float(near_home_ori_offset_max_deg),
            external_ee_target=external_ee_target,
            external_ee_target_source=external_ee_target_source,
            runtime_controlled_indices=runtime_controlled_indices,
            policy_mode="det_student",
            runtime_mode=runtime_mode,
            episodes=int(post_train_eval_episodes),
            steps_per_episode=int(post_train_eval_steps_per_episode),
            ee_pos_success_threshold=float(ee_pos_success_threshold),
            ee_ori_success_threshold=float(ee_ori_success_threshold),
            reset_near_home_eps=float(reset_near_home_eps),
            sac_seed=int(seed),
            gz_world_name=str(gz_world_name),
            gz_target_entity_name=str(gz_target_entity_name),
            gz_target_world_offset_xyz=tuple(float(v) for v in gz_target_world_offset_xyz),
            reward_config=reward_config,
            fixed_eval_suite=fixed_eval_suite,
            eval_name="deterministic_student",
            eval_stochastic=False,
            eval_exploration_std_scale=0.0,
            artifact_key_prefix="student_post_train_eval",
        )
        gap_outputs = _run_gap_diagnosis_gz(
            run_id=run_id,
            artifact_root=artifact_root,
            agent=policy,
            runtime=runtime,
            stage=stage,
            target_mode=target_mode,
            near_home_profile=near_home_profile,
            near_home_pos_offset_min_m=float(near_home_pos_offset_min_m),
            near_home_pos_offset_max_m=float(near_home_pos_offset_max_m),
            near_home_ori_offset_min_deg=float(near_home_ori_offset_min_deg),
            near_home_ori_offset_max_deg=float(near_home_ori_offset_max_deg),
            external_ee_target=external_ee_target,
            external_ee_target_source=external_ee_target_source,
            runtime_controlled_indices=runtime_controlled_indices,
            policy_mode="det_student",
            runtime_mode=runtime_mode,
            episodes=int(gap_eval_episodes if int(gap_eval_episodes) > 0 else post_train_eval_episodes),
            steps_per_episode=int(gap_eval_steps_per_episode if int(gap_eval_steps_per_episode) > 0 else post_train_eval_steps_per_episode),
            ee_pos_success_threshold=float(ee_pos_success_threshold),
            ee_ori_success_threshold=float(ee_ori_success_threshold),
            reset_near_home_eps=float(reset_near_home_eps),
            sac_seed=int(seed),
            gz_world_name=str(gz_world_name),
            gz_target_entity_name=str(gz_target_entity_name),
            gz_target_world_offset_xyz=tuple(float(v) for v in gz_target_world_offset_xyz),
            reward_config=reward_config,
            fixed_eval_suite=fixed_eval_suite,
            eval_specs=gap_eval_specs,
        )
    finally:
        runtime.close()

    student_metrics = dict(eval_outputs["summary"].get("metrics", {}) or {})
    student_gap_metrics = dict(gap_outputs["summary"].get("gap_metrics", {}) or {})
    best_teacher = _best_teacher_metrics(teacher_refs)
    success_criteria = {
        "level1_outer_mean_final": bool(
            float(student_metrics.get("true_outer_hit_rate", 0.0)) > float(best_teacher.get("true_outer_hit_rate", 0.0))
            and float(student_metrics.get("mean_final_dpos", 1.0e9)) < float(best_teacher.get("mean_final_dpos", 1.0e9))
            and float(student_metrics.get("regression_rate", 1.0)) <= float(best_teacher.get("regression_rate", 1.0))
        ),
        "level2_inner_nonzero": bool(float(student_metrics.get("true_inner_hit_rate", 0.0)) > 0.0),
        "level3_success_higher": bool(
            float(student_metrics.get("success_rate", 0.0)) > float(best_teacher.get("success_rate", 0.0))
        ),
    }

    comparison = {
        "student_checkpoint": str(student_checkpoint),
        "student_run_id": str(student_run_id),
        "student_metrics": student_metrics,
        "student_gap_metrics": student_gap_metrics,
        "teacher_references": teacher_refs,
        "best_teacher_metrics": best_teacher,
        "success_criteria": success_criteria,
        "fixed_eval_suite": fixed_eval_suite,
        "reward_profile": reward_profile_resolved,
        "action_scale": float(action_scale),
        "artifacts": {
            **dict(eval_outputs["artifacts"]),
            **dict(gap_outputs["artifacts"]),
        },
    }

    summary_path = artifact_root / "student_eval_summary.json"
    _write_json(summary_path, comparison)

    md_lines = [
        "# Deterministic Student Evaluation",
        "",
        f"- student_checkpoint: `{student_checkpoint}`",
        f"- student_run_id: `{student_run_id}`",
        f"- fixed_eval_suite_id: `{fixed_eval_suite.get('suite_id', '')}`",
        "",
        "## Student Metrics",
        f"- true_outer_hit_rate: `{float(student_metrics.get('true_outer_hit_rate', 0.0)):.4f}`",
        f"- true_inner_hit_rate: `{float(student_metrics.get('true_inner_hit_rate', 0.0)):.4f}`",
        f"- true_dwell_hit_rate: `{float(student_metrics.get('true_dwell_hit_rate', 0.0)):.4f}`",
        f"- true_basin_hit_rate: `{float(student_metrics.get('true_basin_hit_rate', 0.0)):.4f}`",
        f"- mean_final_dpos: `{float(student_metrics.get('mean_final_dpos', 0.0)):.6f}`",
        f"- regression_rate: `{float(student_metrics.get('regression_rate', 0.0)):.4f}`",
        f"- final_action_l2_mean: `{float(student_metrics.get('final_action_l2_mean', 0.0)):.6f}`",
        "",
        "## Teacher Baselines",
    ]
    for teacher in teacher_refs:
        tm = teacher["deterministic_metrics"]
        md_lines.append(
            f"- `{teacher['run_id']}`: outer=`{float(tm.get('true_outer_hit_rate', 0.0)):.4f}`, "
            f"inner=`{float(tm.get('true_inner_hit_rate', 0.0)):.4f}`, "
            f"mean_final_dpos=`{float(tm.get('mean_final_dpos', 0.0)):.6f}`, "
            f"regression=`{float(tm.get('regression_rate', 0.0)):.4f}`"
        )
    md_lines.extend(
        [
            "",
            "## Success Criteria",
            f"- level1_outer_mean_final: `{success_criteria['level1_outer_mean_final']}`",
            f"- level2_inner_nonzero: `{success_criteria['level2_inner_nonzero']}`",
            f"- level3_success_higher: `{success_criteria['level3_success_higher']}`",
            "",
            "## Verdict",
            "- teacher-student deterministic extraction is better only if outer/inner/final retention improve together.",
        ]
    )
    report_path = artifact_root / "student_eval_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "student_eval_summary": str(summary_path),
        "student_eval_report": str(report_path),
        **comparison["artifacts"],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deterministic student policy against teacher baselines")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--runtime-mode", choices=["gz"], default="gz")
    parser.add_argument("--stage-profile", default="s0_b")
    parser.add_argument("--target-mode", default="near_home")
    parser.add_argument("--source-root", default="artifacts/v5_1/e2e")
    parser.add_argument("--compare-runs", default=",".join(DEFAULT_TEACHER_COMPARE_RUNS))
    parser.add_argument("--eval-suite-run", default="")
    parser.add_argument("--runtime-joint-names", required=True)
    parser.add_argument("--trajectory-topic", default="/arm_controller/joint_trajectory")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--reward-profile", default="phase_a_bootstrap_v2")
    parser.add_argument("--external-task-prop", default="tray")
    parser.add_argument("--external-task-src-idx", type=int, default=2)
    parser.add_argument("--external-task-dst-idx", type=int, default=7)
    parser.add_argument("--external-task-waypoint-index", type=int, default=2)
    parser.add_argument("--near-home-profile", default="TC0")
    parser.add_argument("--near-home-pos-offset-min-m", type=float, default=0.15)
    parser.add_argument("--near-home-pos-offset-max-m", type=float, default=0.15)
    parser.add_argument("--near-home-ori-offset-min-deg", type=float, default=0.0)
    parser.add_argument("--near-home-ori-offset-max-deg", type=float, default=0.0)
    parser.add_argument("--gz-world-name", default="empty")
    parser.add_argument("--gz-target-entity-name", default="v5_1_target_marker")
    parser.add_argument("--gz-target-world-offset-x", type=float, default=0.0)
    parser.add_argument("--gz-target-world-offset-y", type=float, default=0.0)
    parser.add_argument("--gz-target-world-offset-z", type=float, default=1.04)
    parser.add_argument("--post-train-eval-episodes", type=int, default=5)
    parser.add_argument("--post-train-eval-steps-per-episode", type=int, default=10)
    parser.add_argument("--gap-eval-scales", default="det,0.10,0.30,0.60")
    parser.add_argument("--gap-eval-episodes", type=int, default=5)
    parser.add_argument("--gap-eval-steps-per-episode", type=int, default=10)
    parser.add_argument("--ee-pos-success-threshold", type=float, default=0.08)
    parser.add_argument("--ee-ori-success-threshold", type=float, default=0.12)
    parser.add_argument("--reset-near-home-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    out = evaluate_deterministic_student(
        student_checkpoint=Path(args.student_checkpoint),
        artifact_root=Path(args.artifact_root),
        runtime_mode=str(args.runtime_mode),
        stage_profile=str(args.stage_profile),
        target_mode=str(args.target_mode),
        source_root=Path(args.source_root),
        compare_runs=parse_csv_list(args.compare_runs, DEFAULT_TEACHER_COMPARE_RUNS),
        eval_suite_run=str(args.eval_suite_run).strip() or None,
        runtime_joint_names=parse_csv_list(args.runtime_joint_names),
        trajectory_topic=str(args.trajectory_topic),
        joint_state_topic=str(args.joint_state_topic),
        reward_profile=str(args.reward_profile),
        external_task_prop=str(args.external_task_prop),
        external_task_src_idx=int(args.external_task_src_idx),
        external_task_dst_idx=int(args.external_task_dst_idx),
        external_task_waypoint_index=int(args.external_task_waypoint_index),
        near_home_profile=str(args.near_home_profile),
        near_home_pos_offset_min_m=float(args.near_home_pos_offset_min_m),
        near_home_pos_offset_max_m=float(args.near_home_pos_offset_max_m),
        near_home_ori_offset_min_deg=float(args.near_home_ori_offset_min_deg),
        near_home_ori_offset_max_deg=float(args.near_home_ori_offset_max_deg),
        gz_world_name=str(args.gz_world_name),
        gz_target_entity_name=str(args.gz_target_entity_name),
        gz_target_world_offset_xyz=(
            float(args.gz_target_world_offset_x),
            float(args.gz_target_world_offset_y),
            float(args.gz_target_world_offset_z),
        ),
        post_train_eval_episodes=int(args.post_train_eval_episodes),
        post_train_eval_steps_per_episode=int(args.post_train_eval_steps_per_episode),
        gap_eval_scales=str(args.gap_eval_scales),
        gap_eval_episodes=int(args.gap_eval_episodes),
        gap_eval_steps_per_episode=int(args.gap_eval_steps_per_episode),
        ee_pos_success_threshold=float(args.ee_pos_success_threshold),
        ee_ori_success_threshold=float(args.ee_ori_success_threshold),
        reset_near_home_eps=float(args.reset_near_home_eps),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
