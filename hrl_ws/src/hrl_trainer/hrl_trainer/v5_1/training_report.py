"""Lightweight V5.1 training report generator.

The pipeline treats reporting as best-effort, but the module itself must exist
so the training entrypoint can import cleanly from source checkouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

_OUTER_SHELL_POS_M = 0.08
_INNER_SHELL_POS_M = 0.04
_DWELL_POS_M = 0.025


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_rate(count: float, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _zone_from_dpos(dpos: float) -> str:
    d = float(dpos)
    if d < _DWELL_POS_M:
        return "dwell"
    if d < _INNER_SHELL_POS_M:
        return "inner"
    if d < _OUTER_SHELL_POS_M:
        return "outer"
    return "outside"


def _episode_true_flag(ep: dict[str, Any], key: str) -> bool:
    if key in ep:
        return bool(ep.get(key, False))
    min_zone = _zone_from_dpos(_as_float(ep.get("min_dpos"), default=1.0))
    final_zone = _zone_from_dpos(_as_float(ep.get("final_dpos"), default=1.0))
    fallback = {
        "true_outer_hit": min_zone == "outer",
        "true_inner_hit": min_zone == "inner",
        "true_dwell_hit": min_zone == "dwell",
        "true_basin_hit": min_zone in {"outer", "inner", "dwell"},
        "true_inner_or_dwell_hit": min_zone in {"inner", "dwell"},
        "true_final_outer": final_zone == "outer",
        "true_final_inner": final_zone == "inner",
        "true_final_dwell": final_zone == "dwell",
        "true_final_basin": final_zone in {"outer", "inner", "dwell"},
        "true_final_inner_or_dwell": final_zone in {"inner", "dwell"},
    }
    return bool(fallback.get(key, False))


def _episode_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "episodes_completed": 0,
            "success_sum": 0,
            "success_rate": 0.0,
            "best_min_dpos": 0.0,
            "mean_min_dpos": 0.0,
            "mean_final_dpos": 0.0,
            "regression_rate": 0.0,
            "max_dwell_max": 0,
        }

    min_dpos = [float(ep.get("min_dpos", 0.0)) for ep in episodes]
    final_dpos = [float(ep.get("final_dpos", 0.0)) for ep in episodes]
    success_sum = int(sum(int(ep.get("success_count", 0)) for ep in episodes))
    regressions = sum(1 for ep in episodes if float(ep.get("final_dpos", 0.0)) > float(ep.get("min_dpos", 0.0)))
    return {
        "episodes_completed": int(len(episodes)),
        "success_sum": success_sum,
        "success_rate": _safe_rate(success_sum, len(episodes)),
        "best_min_dpos": float(min(min_dpos)),
        "mean_min_dpos": _safe_mean(min_dpos),
        "mean_final_dpos": _safe_mean(final_dpos),
        "regression_rate": _safe_rate(regressions, len(episodes)),
        "max_dwell_max": int(max((int(ep.get("max_dwell_count", 0)) for ep in episodes), default=0)),
        "near_goal_entries_sum": int(sum(int(ep.get("near_goal_entry_count", 0)) for ep in episodes)),
        "shell_hit_sum": int(sum(1 for ep in episodes if bool(ep.get("shell_hit", False)))),
        "inner_shell_hit_sum": int(sum(1 for ep in episodes if bool(ep.get("inner_shell_hit", False)))),
        "dwell_hit_sum": int(sum(1 for ep in episodes if bool(ep.get("dwell_hit", False)))),
        "true_outer_hit_sum": int(sum(1 for ep in episodes if _episode_true_flag(ep, "true_outer_hit"))),
        "true_outer_hit_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_outer_hit")), len(episodes)),
        "true_inner_hit_sum": int(sum(1 for ep in episodes if _episode_true_flag(ep, "true_inner_hit"))),
        "true_inner_hit_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_inner_hit")), len(episodes)),
        "true_dwell_hit_sum": int(sum(1 for ep in episodes if _episode_true_flag(ep, "true_dwell_hit"))),
        "true_dwell_hit_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_dwell_hit")), len(episodes)),
        "true_basin_hit_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_basin_hit")), len(episodes)),
        "true_final_outer_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_final_outer")), len(episodes)),
        "true_final_inner_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_final_inner")), len(episodes)),
        "true_final_dwell_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_final_dwell")), len(episodes)),
        "true_final_basin_rate": _safe_rate(sum(1 for ep in episodes if _episode_true_flag(ep, "true_final_basin")), len(episodes)),
        "clamp_sum": int(sum(int(ep.get("clamp_count", 0)) for ep in episodes)),
        "zone_exit_count_sum": int(sum(int(ep.get("zone_exit_count", 0)) for ep in episodes)),
    }


def _best_and_worst(episodes: list[dict[str, Any]], *, n: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = (
        "episode",
        "success_count",
        "total_reward",
        "min_dpos",
        "final_dpos",
        "true_min_zone",
        "true_final_zone",
        "max_dwell_count",
        "near_goal_entry_count",
        "clamp_count",
        "done_reason",
    )

    def compact(ep: dict[str, Any]) -> dict[str, Any]:
        return {k: ep.get(k) for k in fields if k in ep}

    by_best = sorted(episodes, key=lambda ep: float(ep.get("min_dpos", 1.0)))
    by_worst = sorted(episodes, key=lambda ep: float(ep.get("final_dpos", 0.0)), reverse=True)
    return [compact(ep) for ep in by_best[:n]], [compact(ep) for ep in by_worst[:n]]


def _rolling_mean(values: list[float], window: int = 10) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    w = max(1, int(window))
    for idx in range(len(values)):
        start = max(0, idx + 1 - w)
        out.append(_safe_mean(values[start : idx + 1]))
    return out


def _write_plots(artifact_root: Path, plot_dir: Path, episodes: list[dict[str, Any]]) -> dict[str, str]:
    """Best-effort PNG plots for quick run diagnosis."""
    if not episodes:
        return {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    plot_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    xs = [_as_float(ep.get("episode", idx)) for idx, ep in enumerate(episodes)]

    def save_current(name: str) -> None:
        path = plot_dir / f"{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        plots[name] = str(path)

    # Distance quality: closest point vs final point per episode.
    min_dpos = [_as_float(ep.get("min_dpos")) for ep in episodes]
    final_dpos = [_as_float(ep.get("final_dpos")) for ep in episodes]
    plt.figure(figsize=(10, 4.5))
    plt.plot(xs, min_dpos, label="min dpos", linewidth=1.8)
    plt.plot(xs, final_dpos, label="final dpos", linewidth=1.8)
    for y, label, color in (
        (0.08, "outer shell 0.08m", "#7a7a7a"),
        (0.04, "inner shell 0.04m", "#4f7db8"),
        (0.025, "dwell 0.025m", "#3b8f54"),
    ):
        plt.axhline(y, linestyle="--", linewidth=1.0, color=color, alpha=0.65, label=label)
    plt.xlabel("episode")
    plt.ylabel("position error (m)")
    plt.title("Distance To Target")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    save_current("distance_to_target")

    # Rolling true-zone hit and success rates.
    success = [1.0 if int(ep.get("success_count", 0) or 0) > 0 else 0.0 for ep in episodes]
    shell = [1.0 if _episode_true_flag(ep, "true_outer_hit") else 0.0 for ep in episodes]
    inner = [1.0 if _episode_true_flag(ep, "true_inner_hit") else 0.0 for ep in episodes]
    dwell = [1.0 if _episode_true_flag(ep, "true_dwell_hit") else 0.0 for ep in episodes]
    basin = [1.0 if _episode_true_flag(ep, "true_basin_hit") else 0.0 for ep in episodes]
    plt.figure(figsize=(10, 4.5))
    for values, label in (
        (basin, "basin hit"),
        (shell, "outer best-zone"),
        (inner, "inner best-zone"),
        (dwell, "dwell best-zone"),
        (success, "success"),
    ):
        plt.plot(xs, _rolling_mean(values, window=10), label=f"{label} rolling@10", linewidth=1.8)
    plt.ylim(-0.03, 1.03)
    plt.xlabel("episode")
    plt.ylabel("rate")
    plt.title("Rolling Zone Hits And Success")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    save_current("zone_success_rates")

    # Safety and action correction pressure.
    reject_rate = [_as_float(ep.get("reject_rate")) for ep in episodes]
    sum_delta_norm = [_as_float(ep.get("sum_delta_norm")) for ep in episodes]
    clamp_count = [_as_float(ep.get("clamp_count")) for ep in episodes]
    projection_count = [_as_float(ep.get("projection_count")) for ep in episodes]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(xs, reject_rate, label="reject rate", linewidth=1.8)
    axes[0].plot(xs, clamp_count, label="clamp count", linewidth=1.2, alpha=0.8)
    axes[0].plot(xs, projection_count, label="projection count", linewidth=1.2, alpha=0.8)
    axes[0].set_ylabel("count / rate")
    axes[0].set_title("Safety Events")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(xs, sum_delta_norm, label="sum delta_norm", linewidth=1.8, color="#9b5c2e")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("normalized correction")
    axes[1].set_title("Executed-vs-Raw Correction")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    save_current("action_safety")

    # Periodic deterministic eval trajectory, if available.
    curriculum = _read_json(artifact_root / "curriculum_state.json")
    periodic = list(curriculum.get("periodic_eval_history", []) or [])
    if periodic:
        eval_x = [int(rec.get("episode", idx)) + 1 for idx, rec in enumerate(periodic)]
        periodic_metrics = [dict(rec.get("metrics") or {}) for rec in periodic]

        def eval_metric(metrics: dict[str, Any], key: str, fallback_key: str | None = None) -> float:
            if key in metrics:
                return _as_float(metrics.get(key))
            return _as_float(metrics.get(fallback_key)) if fallback_key is not None else 0.0

        det_success = [eval_metric(m, "det_success_rate") for m in periodic_metrics]
        det_basin = [eval_metric(m, "true_basin_hit_rate", "shell_hit_rate") for m in periodic_metrics]
        det_outer = [eval_metric(m, "true_outer_hit_rate", "shell_hit_rate") for m in periodic_metrics]
        det_inner = [eval_metric(m, "true_inner_hit_rate", "inner_shell_hit_rate") for m in periodic_metrics]
        det_dwell = [eval_metric(m, "true_dwell_hit_rate", "dwell_hit_rate") for m in periodic_metrics]
        det_final = [eval_metric(m, "mean_final_dpos") for m in periodic_metrics]
        det_best = [eval_metric(m, "best_min_dpos") for m in periodic_metrics]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for values, label in (
            (det_basin, "det basin"),
            (det_outer, "det outer best-zone"),
            (det_inner, "det inner best-zone"),
            (det_dwell, "det dwell best-zone"),
            (det_success, "det success"),
        ):
            axes[0].plot(eval_x, values, marker="o", linewidth=1.6, label=label)
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].set_ylabel("rate")
        axes[0].set_title("Periodic Deterministic Eval Rates")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)
        axes[1].plot(eval_x, det_best, marker="o", linewidth=1.6, label="best min dpos")
        axes[1].plot(eval_x, det_final, marker="o", linewidth=1.6, label="mean final dpos")
        axes[1].axhline(0.08, linestyle="--", linewidth=1.0, color="#7a7a7a", alpha=0.65)
        axes[1].axhline(0.04, linestyle="--", linewidth=1.0, color="#4f7db8", alpha=0.65)
        axes[1].axhline(0.025, linestyle="--", linewidth=1.0, color="#3b8f54", alpha=0.65)
        axes[1].set_xlabel("training episode")
        axes[1].set_ylabel("position error (m)")
        axes[1].set_title("Periodic Deterministic Eval Distance")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best", fontsize=8)
        save_current("periodic_deterministic_eval")

        target_entropy = [eval_metric(m, "target_entropy") for m in periodic_metrics]
        alpha = [eval_metric(m, "alpha") for m in periodic_metrics]
        action_ratio = [eval_metric(m, "det_action_l2_over_stoch_action_l2") for m in periodic_metrics]
        raw_ratio = [eval_metric(m, "det_raw_norm_over_stoch_raw_norm") for m in periodic_metrics]
        det_action = [eval_metric(m, "det_action_l2_mean") for m in periodic_metrics]
        stoch_action = [eval_metric(m, "stoch_action_l2_mean") for m in periodic_metrics]
        if any(target_entropy) or any(action_ratio) or any(det_action):
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(eval_x, target_entropy, marker="o", linewidth=1.6, label="target entropy")
            axes[0].plot(eval_x, alpha, marker="o", linewidth=1.2, label="alpha")
            axes[0].set_ylabel("value")
            axes[0].set_title("Entropy Annealing State")
            axes[0].grid(alpha=0.25)
            axes[0].legend(loc="best", fontsize=8)
            axes[1].plot(eval_x, det_action, marker="o", linewidth=1.6, label="det action L2")
            axes[1].plot(eval_x, stoch_action, marker="o", linewidth=1.6, label="stoch action L2")
            axes[1].set_ylabel("action L2")
            axes[1].set_title("Mean-vs-Stochastic Action Size")
            axes[1].grid(alpha=0.25)
            axes[1].legend(loc="best", fontsize=8)
            axes[2].plot(eval_x, action_ratio, marker="o", linewidth=1.6, label="det/stoch action L2")
            axes[2].plot(eval_x, raw_ratio, marker="o", linewidth=1.6, label="det/stoch raw norm")
            axes[2].set_xlabel("training episode")
            axes[2].set_ylabel("ratio")
            axes[2].set_title("Stochastic-to-Deterministic Gap")
            axes[2].grid(alpha=0.25)
            axes[2].legend(loc="best", fontsize=8)
            save_current("entropy_annealing")

    gap = _read_json(artifact_root / "eval_gap" / "gap_diagnosis_summary.json")
    gap_records = list(gap.get("records", []) or [])
    if gap_records:
        labels = [str(rec.get("label", idx)) for idx, rec in enumerate(gap_records)]
        gx = list(range(len(labels)))
        success_rate = [_as_float((rec.get("metrics") or {}).get("success_rate")) for rec in gap_records]
        basin_rate = [_as_float((rec.get("metrics") or {}).get("true_basin_hit_rate")) for rec in gap_records]
        inner_rate = [_as_float((rec.get("metrics") or {}).get("true_inner_hit_rate")) for rec in gap_records]
        dwell_rate = [_as_float((rec.get("metrics") or {}).get("true_dwell_hit_rate")) for rec in gap_records]
        final_dpos = [_as_float((rec.get("metrics") or {}).get("mean_final_dpos")) for rec in gap_records]
        action_l2 = [_as_float((rec.get("action_stats") or {}).get("final_action_l2_mean")) for rec in gap_records]
        raw_norm = [_as_float((rec.get("action_stats") or {}).get("raw_norm_mean")) for rec in gap_records]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for values, label in (
            (basin_rate, "basin"),
            (inner_rate, "inner"),
            (dwell_rate, "dwell"),
            (success_rate, "success"),
        ):
            axes[0].plot(gx, values, marker="o", linewidth=1.8, label=label)
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].set_ylabel("rate")
        axes[0].set_title("Noise Sweep: Eval Quality")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)
        axes[1].plot(gx, final_dpos, marker="o", linewidth=1.8, label="mean final dpos")
        axes[1].plot(gx, action_l2, marker="o", linewidth=1.8, label="final action L2")
        axes[1].plot(gx, raw_norm, marker="o", linewidth=1.8, label="raw norm")
        axes[1].set_xticks(gx)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("value")
        axes[1].set_title("Noise Sweep: Action Size")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best", fontsize=8)
        save_current("gap_noise_sweep")

    train_metrics = _read_jsonl(artifact_root / "train_metrics.jsonl")
    if train_metrics:
        tx = list(range(len(train_metrics)))
        good_frac = [_as_float(row.get("distill_good_fraction")) for row in train_metrics]
        good_count = [_as_float(row.get("distill_good_count")) for row in train_metrics]
        quality_mean = [_as_float(row.get("distill_quality_mean")) for row in train_metrics]
        adv_mean = [_as_float(row.get("distill_advantage_mean")) for row in train_metrics]
        mean_action = [_as_float(row.get("distill_mean_action_l2")) for row in train_metrics]
        target_action = [_as_float(row.get("distill_target_action_l2")) for row in train_metrics]
        active_lambda = [_as_float(row.get("active_distill_lambda")) for row in train_metrics]

        det_outer_periodic = []
        det_ratio_periodic = []
        eval_x = []
        if periodic:
            eval_x = [int(rec.get("episode", idx)) + 1 for idx, rec in enumerate(periodic)]
            periodic_metrics = [dict(rec.get("metrics") or {}) for rec in periodic]
            det_outer_periodic = [eval_metric(m, "det_true_outer_hit_rate", "true_outer_hit_rate") for m in periodic_metrics]
            det_ratio_periodic = [eval_metric(m, "det_action_l2_over_stoch_action_l2") for m in periodic_metrics]

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
        axes[0].plot(tx, _rolling_mean(good_frac, window=20), linewidth=1.6, label="distill good fraction")
        axes[0].plot(tx, _rolling_mean(active_lambda, window=20), linewidth=1.2, label="active distill lambda")
        axes[0].set_ylabel("fraction / lambda")
        axes[0].set_title("Solidification Activation")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)

        axes[1].plot(tx, _rolling_mean(good_count, window=20), linewidth=1.6, label="good count")
        axes[1].plot(tx, _rolling_mean(quality_mean, window=20), linewidth=1.6, label="quality mean")
        axes[1].plot(tx, _rolling_mean(adv_mean, window=20), linewidth=1.2, label="advantage mean")
        axes[1].set_ylabel("value")
        axes[1].set_title("Selected Transition Quality")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best", fontsize=8)

        axes[2].plot(tx, _rolling_mean(mean_action, window=20), linewidth=1.6, label="distill mean action L2")
        axes[2].plot(tx, _rolling_mean(target_action, window=20), linewidth=1.6, label="distill target action L2")
        axes[2].set_ylabel("action L2")
        axes[2].set_title("Mean-Policy Extraction")
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best", fontsize=8)

        if eval_x:
            axes[3].plot(eval_x, det_ratio_periodic, marker="o", linewidth=1.6, label="det/stoch action ratio")
            axes[3].plot(eval_x, det_outer_periodic, marker="o", linewidth=1.6, label="det true outer hit")
        axes[3].set_xlabel("training step / periodic eval episode")
        axes[3].set_ylabel("rate / ratio")
        axes[3].set_title("Deterministic Solidification Outcome")
        axes[3].grid(alpha=0.25)
        if eval_x:
            axes[3].legend(loc="best", fontsize=8)
        save_current("solidification_metrics")

    return plots


def _gap_diagnosis(artifact_root: Path) -> dict[str, Any]:
    summary = _read_json(artifact_root / "eval_gap" / "gap_diagnosis_summary.json")
    if not summary:
        return {"available": False}
    records: list[dict[str, Any]] = []
    for rec in list(summary.get("records", []) or []):
        metrics = dict(rec.get("metrics") or {})
        action_stats = dict(rec.get("action_stats") or {})
        records.append(
            {
                "label": str(rec.get("label", "")),
                "stochastic": bool(rec.get("stochastic", False)),
                "exploration_std_scale": float(rec.get("exploration_std_scale", 0.0)),
                "success_rate": float(metrics.get("success_rate", 0.0)),
                "true_basin_hit_rate": float(metrics.get("true_basin_hit_rate", 0.0)),
                "true_inner_hit_rate": float(metrics.get("true_inner_hit_rate", 0.0)),
                "true_dwell_hit_rate": float(metrics.get("true_dwell_hit_rate", 0.0)),
                "mean_final_dpos": float(metrics.get("mean_final_dpos", 0.0)),
                "mean_final_minus_min": float(metrics.get("mean_final_minus_min", 0.0)),
                "final_action_l2_mean": float(action_stats.get("final_action_l2_mean", 0.0)),
                "raw_norm_mean": float(action_stats.get("raw_norm_mean", 0.0)),
                "exec_norm_mean": float(action_stats.get("exec_norm_mean", 0.0)),
                "delta_norm_mean": float(action_stats.get("delta_norm_mean", 0.0)),
            }
        )
    return {
        "available": True,
        "summary_path": str(artifact_root / "eval_gap" / "gap_diagnosis_summary.json"),
        "gap_metrics": dict(summary.get("gap_metrics", {}) or {}),
        "records": records,
    }


def _train_metrics(artifact_root: Path) -> dict[str, Any]:
    rows = _read_jsonl(artifact_root / "train_metrics.jsonl")
    if not rows:
        return {"available": False, "rows": []}
    distilled = []
    for row in rows:
        distilled.append(
            {
                "episode": int(row.get("episode", 0)),
                "step": int(row.get("step", 0)),
                "entropy_stage": str(row.get("entropy_stage", "")),
                "target_entropy": float(row.get("target_entropy", 0.0)),
                "active_distill_lambda": float(row.get("active_distill_lambda", 0.0)),
                "distill_enabled": float(row.get("distill_enabled", 0.0)),
                "distill_triggered": float(row.get("distill_triggered", 0.0)),
                "distill_loss": float(row.get("distill_loss", 0.0)),
                "distill_good_count": float(row.get("distill_good_count", 0.0)),
                "distill_good_fraction": float(row.get("distill_good_fraction", 0.0)),
                "distill_quality_mean": float(row.get("distill_quality_mean", 0.0)),
                "distill_advantage_mean": float(row.get("distill_advantage_mean", 0.0)),
                "distill_mean_action_l2": float(row.get("distill_mean_action_l2", 0.0)),
                "distill_target_action_l2": float(row.get("distill_target_action_l2", 0.0)),
            }
        )
    return {"available": True, "rows": distilled}


def _deterministic_eval(artifact_root: Path, train_stats: dict[str, Any]) -> dict[str, Any]:
    eval_summary = _read_json(artifact_root / "eval" / "deterministic_summary.json")
    metrics = dict(eval_summary.get("metrics", {}))
    if not metrics:
        return {"available": False}
    return {
        "available": True,
        "episodes_completed": int(metrics.get("episodes_completed", 0)),
        "success_rate": float(metrics.get("success_rate", 0.0)),
        "best_min_dpos": float(metrics.get("best_min_dpos", 0.0)),
        "mean_min_dpos": float(metrics.get("mean_min_dpos", 0.0)),
        "mean_final_dpos": float(metrics.get("mean_final_dpos", 0.0)),
        "regression_rate": float(metrics.get("regression_rate", 0.0)),
        "max_dwell_max": int(metrics.get("max_dwell_max", 0)),
        "clamp_mean": float(metrics.get("clamp_mean", 0.0)),
        "final_action_l2_mean": float(metrics.get("final_action_l2_mean", 0.0)),
        "raw_norm_mean": float(metrics.get("raw_norm_mean", 0.0)),
        "min_dpos_improvement_vs_train": float(train_stats.get("mean_min_dpos", 0.0))
        - float(metrics.get("mean_min_dpos", 0.0)),
        "final_dpos_improvement_vs_train": float(train_stats.get("mean_final_dpos", 0.0))
        - float(metrics.get("mean_final_dpos", 0.0)),
    }


def _markdown(report: dict[str, Any]) -> str:
    headline = report["headline"]
    episode_stats = report["episode_stats"]
    det = report["deterministic_eval"]
    lines = [
        f"# Training Report: {report['run_id']}",
        "",
        "## Headline",
        f"- Episodes completed: {headline['episodes_completed']}",
        f"- Train success rate: {headline['success_rate']:.3f}",
        f"- Best min dpos: {episode_stats['best_min_dpos']:.4f}",
        f"- Mean final dpos: {episode_stats['mean_final_dpos']:.4f}",
        f"- Regression rate: {episode_stats['regression_rate']:.3f}",
        f"- True basin hit rate: {episode_stats.get('true_basin_hit_rate', 0.0):.3f}",
        f"- True outer / inner / dwell rates: "
        f"{episode_stats.get('true_outer_hit_rate', 0.0):.3f} / "
        f"{episode_stats.get('true_inner_hit_rate', 0.0):.3f} / "
        f"{episode_stats.get('true_dwell_hit_rate', 0.0):.3f}",
        f"- True final basin rate: {episode_stats.get('true_final_basin_rate', 0.0):.3f}",
    ]
    if det.get("available"):
        lines.extend(
            [
                "",
                "## Deterministic Eval",
                f"- Episodes completed: {det['episodes_completed']}",
                f"- Success rate: {det['success_rate']:.3f}",
                f"- Mean final dpos: {det['mean_final_dpos']:.4f}",
                f"- Regression rate: {det['regression_rate']:.3f}",
            ]
        )
    gap = dict(report.get("gap_diagnosis", {}) or {})
    if gap.get("available"):
        lines.extend(["", "## Gap Diagnosis"])
        for rec in gap.get("records", []):
            lines.append(
                "- "
                f"{rec['label']}: success={rec['success_rate']:.3f}, "
                f"basin={rec['true_basin_hit_rate']:.3f}, "
                f"inner={rec['true_inner_hit_rate']:.3f}, "
                f"dwell={rec['true_dwell_hit_rate']:.3f}, "
                f"final_dpos={rec['mean_final_dpos']:.4f}, "
                f"action_l2={rec['final_action_l2_mean']:.4f}, "
                f"raw_norm={rec['raw_norm_mean']:.4f}"
            )
    solid = dict(report.get("solidification", {}) or {})
    if solid.get("available"):
        tail = list(solid.get("rows", []) or [])[-1:] or []
        lines.extend(["", "## Solidification"])
        if tail:
            row = tail[0]
            lines.extend(
                [
                    f"- Active distill lambda: {row['active_distill_lambda']:.4f}",
                    f"- Distill good fraction: {row['distill_good_fraction']:.4f}",
                    f"- Distill quality mean: {row['distill_quality_mean']:.4f}",
                    f"- Distill mean/target action L2: {row['distill_mean_action_l2']:.4f} / {row['distill_target_action_l2']:.4f}",
                    f"- Distill advantage mean: {row['distill_advantage_mean']:.4f}",
                ]
            )
    entropy = dict(report.get("entropy_annealing", {}) or {})
    if entropy.get("enabled"):
        current = dict(entropy.get("current_stage", {}) or {})
        state = dict(entropy.get("state", {}) or {})
        history = list(entropy.get("history", []) or [])
        lines.extend(
            [
                "",
                "## Entropy Annealing",
                f"- Mode: {state.get('mode', 'off')}",
                f"- Current stage: {current.get('name', 'unknown')}",
                f"- Current target entropy: {float(current.get('target_entropy', 0.0)):.3f}",
                f"- Stage switches: {len(history)}",
            ]
        )
        for event in history[-5:]:
            lines.append(
                "- "
                f"ep{int(event.get('episode_completed', int(event.get('episode', -1)) + 1))}: "
                f"{event.get('stage_before', '?')}->{event.get('stage_after', '?')}, "
                f"target={float(event.get('target_entropy_after', 0.0)):.3f}, "
                f"reason={event.get('reason', '')}"
            )
    plots = dict(report.get("plots", {}) or {})
    if plots:
        lines.extend(["", "## Plots"])
        for name, path in plots.items():
            rel = Path(path).name
            label = name.replace("_", " ").title()
            lines.append(f"- [{label}](plots/{rel})")
    lines.extend(
        [
            "",
            "## Observations",
            *[f"- {item}" for item in report.get("observations", [])],
            "",
        ]
    )
    return "\n".join(lines)


def write_training_report(artifact_root: str | Path) -> dict[str, str]:
    root = Path(artifact_root)
    analysis_dir = root / "analysis"
    plot_dir = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(root / "pipeline_summary.json")
    episodes = _read_jsonl(root / "episode_reward_summary.jsonl")
    episode_stats = _episode_stats(episodes)
    best_episodes, worst_episodes = _best_and_worst(episodes)
    deterministic_eval = _deterministic_eval(root, episode_stats)
    gap_diagnosis = _gap_diagnosis(root)
    solidification = _train_metrics(root)
    plots = _write_plots(root, plot_dir, episodes)

    observations: list[str] = []
    if episode_stats["success_sum"] > 0:
        observations.append(f"Run produced {episode_stats['success_sum']} successful episodes.")
    else:
        observations.append("No successful training episodes were recorded.")
    if episode_stats["regression_rate"] > 0.5:
        observations.append("Most episodes ended farther from the target than their closest point.")
    if deterministic_eval.get("available"):
        observations.append("Post-train deterministic evaluation is available.")

    metrics = dict(summary.get("metrics", {}))
    report = {
        "artifact_root": str(root),
        "run_id": str(summary.get("run_id", root.name)),
        "headline": {
            "episodes_completed": int(episode_stats["episodes_completed"]),
            "success_rate": float(episode_stats["success_rate"]),
            "reward_mean": float(metrics.get("reward_mean", 0.0)),
            "reward_min": float(metrics.get("reward_min", 0.0)),
            "reward_max": float(metrics.get("reward_max", 0.0)),
            "updates_applied": float(metrics.get("updates_applied", 0.0)),
            "near_goal_pos_m": float(metrics.get("near_goal_pos_m", 0.0)),
            "dwell_pos_m": float(metrics.get("dwell_pos_m", 0.0)),
        },
        "episode_stats": episode_stats,
        "deterministic_eval": deterministic_eval,
        "gap_diagnosis": gap_diagnosis,
        "solidification": solidification,
        "entropy_annealing": dict(summary.get("entropy_annealing", {}) or {}),
        "best_episodes": best_episodes,
        "worst_episodes": worst_episodes,
        "plots": plots,
        "observations": observations,
        "source_files": {
            "summary": str(root / "pipeline_summary.json"),
            "episode_reward_summary": str(root / "episode_reward_summary.jsonl"),
            "post_train_eval_summary": str(root / "eval" / "deterministic_summary.json"),
            "post_train_eval_episode_summary": str(root / "eval" / "deterministic_episode_summary.jsonl"),
            "reward_trace": str(root / "reward_trace.jsonl"),
            "train_metrics": str(root / "train_metrics.jsonl"),
        },
    }

    report_json = analysis_dir / "training_report.json"
    report_md = analysis_dir / "training_report.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(_markdown(report), encoding="utf-8")

    return {
        "training_report_json": str(report_json),
        "training_report_md": str(report_md),
        "training_report_plot_dir": str(plot_dir),
    }
