"""Build deterministic teacher-student extraction datasets from SAC artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any

import numpy as np

from .deterministic_student import DEFAULT_DATASET_SOURCE_RUNS, parse_csv_list
from .pipeline_e2e import _dpos_zone, _obs_from_state, _reward_config_for_profile


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _zone_flags(true_zone: str) -> dict[str, Any]:
    zone = str(true_zone)
    return {
        "true_zone": zone,
        "outside": zone == "outside",
        "outer": zone == "outer",
        "inner": zone == "inner",
        "dwell": zone == "dwell",
        "is_outer": zone == "outer",
        "is_inner": zone == "inner",
        "is_dwell": zone == "dwell",
    }


def _optional_plot_counts(plot_path: Path, *, tier_counts: Counter[str], zone_counts: Counter[str], zone_weight_sums: dict[str, float]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    tier_labels = list(tier_counts.keys()) or ["none"]
    tier_values = [float(tier_counts.get(label, 0.0)) for label in tier_labels]
    axes[0].bar(tier_labels, tier_values, color=["#0f766e", "#2563eb", "#64748b"][: len(tier_labels)])
    axes[0].set_title("Tier Counts")

    zone_labels = ["outside", "outer", "inner", "dwell"]
    zone_values = [float(zone_counts.get(label, 0.0)) for label in zone_labels]
    axes[1].bar(zone_labels, zone_values, color="#f59e0b")
    axes[1].set_title("Zone Counts")

    weight_values = [float(zone_weight_sums.get(label, 0.0)) for label in zone_labels]
    axes[2].bar(zone_labels, weight_values, color="#7c3aed")
    axes[2].set_title("Zone Weight Totals")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return str(plot_path)


def _quality_and_tier(
    *,
    next_dpos: float,
    progress: float,
    true_zone: str,
    done_reason: str,
    success_awarded: bool,
    final_dpos: float,
    final_minus_min: float,
    true_final_basin: bool,
    rejected: bool,
    clamped: bool,
    projected: bool,
    delta_norm: float,
    outer_pos_m: float,
    inner_pos_m: float,
    dwell_pos_m: float,
    support_dpos_m: float,
    min_progress_m: float,
    max_delta_norm: float,
    elite_retention_max: float,
    strong_retention_max: float,
    discard_retention_max: float,
    outer_final_dpos_max: float,
) -> tuple[str | None, float, float, list[str]]:
    reasons: list[str] = []
    is_success = bool(done_reason == "success" or success_awarded)
    is_dwell = str(true_zone) == "dwell"
    is_inner = str(true_zone) == "inner"
    is_outer = str(true_zone) == "outer"
    progress_ok = float(progress) >= float(min_progress_m)
    support_ok = is_outer and progress_ok and float(next_dpos) <= float(support_dpos_m)
    elite_retention = bool(true_final_basin and float(final_minus_min) <= float(elite_retention_max))
    strong_retention = bool(
        float(final_minus_min) <= float(strong_retention_max)
        and (bool(true_final_basin) or float(final_dpos) <= float(outer_final_dpos_max))
    )
    poor_retention = float(final_minus_min) > float(discard_retention_max)

    if bool(rejected):
        return None, 0.0, 0.0, ["rejected"]
    if bool(clamped):
        return None, 0.0, 0.0, ["clamped"]
    if bool(projected):
        return None, 0.0, 0.0, ["projected"]
    if float(delta_norm) > float(max_delta_norm):
        return None, 0.0, 0.0, ["delta_norm_too_large"]
    if poor_retention and not (is_success or is_dwell or is_inner):
        return None, 0.0, 0.0, ["poor_retention"]

    elite = bool(is_success or is_dwell or is_inner or elite_retention)
    strong = bool(
        not elite
        and (
            (is_outer and strong_retention)
            or support_ok
        )
    )
    if not elite and not strong:
        return None, 0.0, 0.0, ["below_priority_threshold"]

    tier = "elite" if elite else "strong"
    if is_success:
        reasons.append("success")
    if is_dwell:
        reasons.append("dwell")
    if is_inner:
        reasons.append("inner")
    if elite_retention:
        reasons.append("elite_retention")
    if strong_retention and is_outer:
        reasons.append("outer_good_retention")
    if support_ok:
        reasons.append("support_progress")

    outer_span = max(float(outer_pos_m) - float(inner_pos_m), 1e-6)
    support_span = max(float(support_dpos_m) - float(inner_pos_m), 1e-6)
    depth_outer = float(np.clip((float(outer_pos_m) - float(next_dpos)) / outer_span, 0.0, 1.0))
    depth_support = float(np.clip((float(support_dpos_m) - float(next_dpos)) / support_span, 0.0, 1.0))
    depth_inner = float(np.clip((float(inner_pos_m) - float(next_dpos)) / max(float(inner_pos_m) - float(dwell_pos_m), 1e-6), 0.0, 1.0))
    progress_score = float(np.clip(float(progress) / max(float(min_progress_m), 1e-6), 0.0, 1.0))
    retention_bonus = 0.0
    if float(final_minus_min) <= float(strong_retention_max):
        retention_bonus += 0.75
    if elite_retention:
        retention_bonus += 1.0
    final_basin_bonus = 0.5 if bool(true_final_basin) else 0.0

    quality = 0.0
    if is_success:
        quality = max(quality, 8.0)
    if is_dwell:
        quality = max(quality, 6.0 + 0.5 * depth_inner)
    if is_inner:
        quality = max(quality, 4.0 + 0.75 * depth_inner)
    if is_outer and strong_retention:
        quality = max(quality, 2.5 + 0.5 * depth_outer)
    if support_ok:
        quality = max(quality, 2.0 + 0.75 * depth_support)
    quality += 0.25 * progress_score + retention_bonus + final_basin_bonus

    sample_weight = max(0.25, quality)
    if tier == "elite":
        sample_weight *= 1.2

    return tier, float(quality), float(sample_weight), reasons


def build_teacher_dataset(
    *,
    source_root: Path,
    source_runs: list[str],
    artifact_root: Path,
    max_delta_norm: float = 0.35,
    elite_retention_max: float = 0.02,
    strong_retention_max: float = 0.04,
    discard_retention_max: float = 0.08,
    outer_support_dpos_m: float = 0.07,
    min_progress_m: float = 0.003,
    outer_final_dpos_max: float = 0.12,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_rows: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    zone_counts: Counter[str] = Counter()
    per_run_counts: Counter[str] = Counter()
    zone_weight_sums: defaultdict[str, float] = defaultdict(float)
    discard_counts: Counter[str] = Counter()
    source_infos: list[dict[str, Any]] = []

    for run_id in source_runs:
        run_root = Path(source_root) / str(run_id)
        pipeline_summary_path = run_root / "pipeline_summary.json"
        reward_trace_path = run_root / "reward_trace.jsonl"
        runtime_trace_path = run_root / "runtime_trace.jsonl"
        episode_summary_path = run_root / "episode_reward_summary.jsonl"
        if not pipeline_summary_path.exists():
            raise FileNotFoundError(f"missing pipeline summary for run {run_id}: {pipeline_summary_path}")
        pipeline_summary = _load_json(pipeline_summary_path)
        reward_profile = str(pipeline_summary.get("reward_profile", "default"))
        reward_config_dict = dict(pipeline_summary.get("reward_config", {}) or {})
        reward_config = _reward_config_for_profile(
            reward_profile,
            action_scale=float(pipeline_summary.get("action_scale", 0.08)),
        )
        if reward_config_dict:
            outer_pos_m = float(reward_config_dict.get("outer_shell_pos_m", reward_config.outer_shell_pos_m))
            inner_pos_m = float(reward_config_dict.get("inner_shell_pos_m", reward_config.inner_shell_pos_m))
            dwell_pos_m = float(reward_config_dict.get("dwell_pos_m", reward_config.dwell_pos_m))
        else:
            outer_pos_m = float(reward_config.outer_shell_pos_m)
            inner_pos_m = float(reward_config.inner_shell_pos_m)
            dwell_pos_m = float(reward_config.dwell_pos_m)

        reward_rows = _load_jsonl(reward_trace_path)
        runtime_rows = _load_jsonl(runtime_trace_path)
        episode_rows = _load_jsonl(episode_summary_path)
        runtime_by_key = {
            (int(row.get("episode", -1)), int(row.get("step", -1))): row
            for row in runtime_rows
            if int(row.get("step", -1)) >= 0
        }
        episode_by_index = {int(row.get("episode", -1)): row for row in episode_rows}
        reward_rows_sorted = sorted(
            [row for row in reward_rows if int(row.get("step", -1)) >= 0],
            key=lambda row: (int(row.get("episode", -1)), int(row.get("step", -1))),
        )

        source_infos.append(
            {
                "run_id": str(run_id),
                "reward_profile": reward_profile,
                "action_scale": float(pipeline_summary.get("action_scale", 0.08)),
                "fixed_eval_suite_id": str((pipeline_summary.get("fixed_eval_suite") or {}).get("suite_id", "")),
                "reward_trace_path": str(reward_trace_path),
                "runtime_trace_path": str(runtime_trace_path),
                "episode_summary_path": str(episode_summary_path),
            }
        )

        prev_q_by_episode: dict[int, np.ndarray] = {}
        prev_action_by_episode: dict[int, np.ndarray] = {}

        for reward_row in reward_rows_sorted:
            episode = int(reward_row.get("episode", -1))
            step = int(reward_row.get("step", -1))
            episode_summary = episode_by_index.get(episode)
            runtime_row = runtime_by_key.get((episode, step))
            if episode_summary is None or runtime_row is None:
                discard_counts["missing_join"] += 1
                continue

            q_now = np.asarray(runtime_row.get("readback_q_before", []), dtype=np.float32)
            q_next = np.asarray(runtime_row.get("readback_q_after", []), dtype=np.float32)
            action_exec = np.asarray(reward_row.get("action_exec", []), dtype=np.float32)
            if q_now.size == 0 or q_next.size == 0 or action_exec.size == 0:
                discard_counts["missing_arrays"] += 1
                continue

            prev_q = prev_q_by_episode.get(episode)
            prev_action = prev_action_by_episode.get(episode)
            dq_now = (q_now - prev_q) if prev_q is not None and prev_q.shape == q_now.shape else np.zeros_like(q_now)
            prev_action_now = prev_action.copy() if prev_action is not None and prev_action.shape == action_exec.shape else np.zeros_like(action_exec)

            ee_target = np.asarray(reward_row.get("ee_target", np.zeros(6, dtype=float)), dtype=np.float32)
            ee_pose = np.asarray(reward_row.get("ee_pose", np.zeros(6, dtype=float)), dtype=np.float32)
            prev_ee_pos_err = ee_target[:3] - ee_pose[:3]
            prev_ee_ori_err = ee_target[3:6] - ee_pose[3:6]
            next_ee_pos_err = np.asarray(reward_row.get("ee_pos_err", prev_ee_pos_err), dtype=np.float32)
            next_ee_ori_err = np.asarray(reward_row.get("ee_ori_err", prev_ee_ori_err), dtype=np.float32)

            obs = _obs_from_state(
                q=q_now,
                dq=dq_now,
                ee_pose_err=np.concatenate([prev_ee_pos_err, prev_ee_ori_err]),
                prev_action=prev_action_now,
            ).astype(np.float32)
            next_obs = _obs_from_state(
                q=q_next,
                dq=(q_next - q_now).astype(np.float32),
                ee_pose_err=np.concatenate([next_ee_pos_err, next_ee_ori_err]),
                prev_action=action_exec,
            ).astype(np.float32)

            prev_dpos = float(np.linalg.norm(prev_ee_pos_err))
            next_dpos = float(np.linalg.norm(next_ee_pos_err))
            progress = float(prev_dpos - next_dpos)
            delta_norm = float(reward_row.get("delta_norm", runtime_row.get("delta_norm", 0.0)))
            done_reason = str(reward_row.get("done_reason", episode_summary.get("done_reason", "running")))
            reward_state_out = dict(reward_row.get("reward_state_out", {}) or {})
            success_awarded = bool(reward_state_out.get("success_awarded", False)) or bool(
                (reward_row.get("components", {}) or {}).get("success_triggered_by_dwell", 0.0)
            )
            final_dpos = float(episode_summary.get("final_dpos", 0.0))
            min_dpos = float(episode_summary.get("min_dpos", 0.0))
            final_minus_min = float(episode_summary.get("final_dpos_minus_min_dpos", final_dpos - min_dpos))
            true_final_basin = bool(episode_summary.get("true_final_basin", False))
            true_zone = _dpos_zone(next_dpos, reward_config)
            clamped = bool(reward_row.get("clamp_triggered", False))
            projected = bool(reward_row.get("projection_triggered", False))
            rejected = bool(reward_row.get("rejected", False))

            tier, quality_score, sample_weight, reasons = _quality_and_tier(
                next_dpos=next_dpos,
                progress=progress,
                true_zone=true_zone,
                done_reason=done_reason,
                success_awarded=success_awarded,
                final_dpos=final_dpos,
                final_minus_min=final_minus_min,
                true_final_basin=true_final_basin,
                rejected=rejected,
                clamped=clamped,
                projected=projected,
                delta_norm=delta_norm,
                outer_pos_m=outer_pos_m,
                inner_pos_m=inner_pos_m,
                dwell_pos_m=dwell_pos_m,
                support_dpos_m=float(outer_support_dpos_m),
                min_progress_m=float(min_progress_m),
                max_delta_norm=float(max_delta_norm),
                elite_retention_max=float(elite_retention_max),
                strong_retention_max=float(strong_retention_max),
                discard_retention_max=float(discard_retention_max),
                outer_final_dpos_max=float(outer_final_dpos_max),
            )
            if tier is None:
                discard_counts[reasons[0] if reasons else "discarded"] += 1
                prev_q_by_episode[episode] = q_now.copy()
                prev_action_by_episode[episode] = action_exec.copy()
                continue

            zone_meta = _zone_flags(true_zone)
            row = {
                "obs": obs.tolist(),
                "next_obs": next_obs.tolist(),
                "target_action_exec": action_exec.tolist(),
                "target_action_raw": list(np.asarray(reward_row.get("action_raw", action_exec), dtype=float)),
                "source_run_id": str(run_id),
                "episode": int(episode),
                "episode_id": str(reward_row.get("episode_id", episode_summary.get("episode_id", f"{run_id}_ep{episode:03d}"))),
                "step": int(step),
                "min_dpos_context": float(min_dpos),
                "final_dpos_context": float(final_dpos),
                "final_dpos_minus_min_dpos": float(final_minus_min),
                "prev_dpos": float(prev_dpos),
                "next_dpos": float(next_dpos),
                "progress": float(progress),
                "quality_score": float(quality_score),
                "sample_weight": float(sample_weight),
                "tier": str(tier),
                "tier_reasons": reasons,
                "delta_norm": float(delta_norm),
                "rejected": bool(rejected),
                "clamped": bool(clamped),
                "projected": bool(projected),
                "is_success": bool(done_reason == "success" or success_awarded),
                "done_reason": str(done_reason),
                "true_final_basin": bool(true_final_basin),
                "source_fixed_eval_suite_id": str((pipeline_summary.get("fixed_eval_suite") or {}).get("suite_id", "")),
                **zone_meta,
            }
            dataset_rows.append(row)
            tier_counts[str(tier)] += 1
            zone_counts[str(true_zone)] += 1
            per_run_counts[str(run_id)] += 1
            zone_weight_sums[str(true_zone)] += float(sample_weight)

            prev_q_by_episode[episode] = q_now.copy()
            prev_action_by_episode[episode] = action_exec.copy()

    dataset_rows.sort(key=lambda row: (row["source_run_id"], int(row["episode"]), int(row["step"])))
    dataset_path = artifact_root / "dataset.jsonl"
    _write_jsonl(dataset_path, dataset_rows)

    summary = {
        "dataset_path": str(dataset_path),
        "total_samples": int(len(dataset_rows)),
        "source_runs": source_infos,
        "obs_dim": int(len(dataset_rows[0]["obs"])) if dataset_rows else 0,
        "action_dim": int(len(dataset_rows[0]["target_action_exec"])) if dataset_rows else 0,
        "tier_counts": dict(tier_counts),
        "zone_counts": dict(zone_counts),
        "zone_weight_sums": {k: float(v) for k, v in zone_weight_sums.items()},
        "per_run_counts": dict(per_run_counts),
        "discard_counts": dict(discard_counts),
        "filters": {
            "max_delta_norm": float(max_delta_norm),
            "elite_retention_max": float(elite_retention_max),
            "strong_retention_max": float(strong_retention_max),
            "discard_retention_max": float(discard_retention_max),
            "outer_support_dpos_m": float(outer_support_dpos_m),
            "min_progress_m": float(min_progress_m),
            "outer_final_dpos_max": float(outer_final_dpos_max),
        },
        "sample_weight_sum": float(sum(float(row["sample_weight"]) for row in dataset_rows)),
        "quality_score_mean": float(np.mean([float(row["quality_score"]) for row in dataset_rows])) if dataset_rows else 0.0,
        "quality_score_max": float(np.max([float(row["quality_score"]) for row in dataset_rows])) if dataset_rows else 0.0,
    }
    summary_path = artifact_root / "dataset_summary.json"
    _save_json(summary_path, summary)

    plot_path = _optional_plot_counts(
        artifact_root / "plots" / "dataset_counts.png",
        tier_counts=tier_counts,
        zone_counts=zone_counts,
        zone_weight_sums={k: float(v) for k, v in zone_weight_sums.items()},
    )
    if plot_path is not None:
        summary["dataset_counts_plot"] = plot_path
        _save_json(summary_path, summary)

    md_lines = [
        "# Teacher Dataset",
        "",
        f"- total_samples: `{summary['total_samples']}`",
        f"- obs_dim: `{summary['obs_dim']}`",
        f"- action_dim: `{summary['action_dim']}`",
        "",
        "## Sources",
    ]
    for source in source_infos:
        md_lines.append(
            f"- `{source['run_id']}`: reward_profile=`{source['reward_profile']}`, "
            f"suite_id=`{source['fixed_eval_suite_id']}`"
        )
    md_lines.extend(["", "## Tier Counts"])
    for key, value in tier_counts.items():
        md_lines.append(f"- `{key}`: `{int(value)}`")
    md_lines.extend(["", "## Zone Counts"])
    for key in ["outside", "outer", "inner", "dwell"]:
        md_lines.append(f"- `{key}`: `{int(zone_counts.get(key, 0))}`")
    md_lines.extend(["", "## Zone Weight Totals"])
    for key in ["outside", "outer", "inner", "dwell"]:
        md_lines.append(f"- `{key}`: `{float(zone_weight_sums.get(key, 0.0)):.4f}`")
    md_lines.extend(["", "## Discard Counts"])
    for key, value in sorted(discard_counts.items()):
        md_lines.append(f"- `{key}`: `{int(value)}`")
    (artifact_root / "dataset_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "dataset_path": str(dataset_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic teacher dataset from teacher runs")
    parser.add_argument("--source-root", default="artifacts/v5_1/e2e")
    parser.add_argument("--source-runs", default=",".join(DEFAULT_DATASET_SOURCE_RUNS))
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--max-delta-norm", type=float, default=0.35)
    parser.add_argument("--elite-retention-max", type=float, default=0.02)
    parser.add_argument("--strong-retention-max", type=float, default=0.04)
    parser.add_argument("--discard-retention-max", type=float, default=0.08)
    parser.add_argument("--outer-support-dpos-m", type=float, default=0.07)
    parser.add_argument("--min-progress-m", type=float, default=0.003)
    parser.add_argument("--outer-final-dpos-max", type=float, default=0.12)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    out = build_teacher_dataset(
        source_root=Path(args.source_root),
        source_runs=parse_csv_list(args.source_runs, DEFAULT_DATASET_SOURCE_RUNS),
        artifact_root=Path(args.artifact_root),
        max_delta_norm=float(args.max_delta_norm),
        elite_retention_max=float(args.elite_retention_max),
        strong_retention_max=float(args.strong_retention_max),
        discard_retention_max=float(args.discard_retention_max),
        outer_support_dpos_m=float(args.outer_support_dpos_m),
        min_progress_m=float(args.min_progress_m),
        outer_final_dpos_max=float(args.outer_final_dpos_max),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
