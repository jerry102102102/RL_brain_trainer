#!/usr/bin/env python3
"""Generate plots for workspace expansion runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    summary = _load(run_dir / "workspace_eval_summary.json") or _load(run_dir / "final_eval/workspace_eval_summary.json")
    stage_metrics = summary.get("stage_metrics") or _load(run_dir / "stage_metrics.json")
    if not stage_metrics:
        print(f"WARN: no stage metrics found under {run_dir}")
        return 1

    stages = sorted(int(k) for k in stage_metrics.keys())
    success = [stage_metrics[str(s)]["success_rate"] if str(s) in stage_metrics else stage_metrics[s]["success_rate"] for s in stages]
    pos = [stage_metrics[str(s)]["mean_final_position_error"] if str(s) in stage_metrics else stage_metrics[s]["mean_final_position_error"] for s in stages]
    ori = [stage_metrics[str(s)]["mean_final_orientation_error"] if str(s) in stage_metrics else stage_metrics[s]["mean_final_orientation_error"] for s in stages]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([str(s) for s in stages], success, color="#2a9d8f")
    ax.axhline(0.85, color="#e76f51", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Workspace stage")
    ax.set_ylabel("Success rate")
    ax.set_title("Workspace Expansion Success by Stage")
    fig.tight_layout()
    fig.savefig(plots / "success_by_stage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(stages, [p * 1000.0 for p in pos], marker="o", label="position error (mm)")
    ax.plot(stages, ori, marker="o", label="orientation error (rad)")
    ax.set_xlabel("Workspace stage")
    ax.set_title("Final Error by Stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "final_error_by_stage.png", dpi=180)
    plt.close(fig)

    rows = summary.get("target_rows", [])
    if rows:
        xs = [r["goal_position"][0] for r in rows]
        ys = [r["goal_position"][1] for r in rows]
        colors = ["#2a9d8f" if r.get("success") else "#e76f51" for r in rows]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(xs, ys, c=colors, s=14, alpha=0.75)
        ax.set_xlabel("target x")
        ax.set_ylabel("target y")
        ax.set_title("Workspace Target Cloud: Success vs Failure")
        fig.tight_layout()
        fig.savefig(plots / "workspace_target_cloud_success_failure.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        reason_counts: dict[str, int] = {}
        for r in rows:
            reason_counts[r.get("failure_reason", "unknown")] = reason_counts.get(r.get("failure_reason", "unknown"), 0) + 1
        ax.bar(reason_counts.keys(), reason_counts.values(), color="#457b9d")
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Failure Reason Counts")
        fig.tight_layout()
        fig.savefig(plots / "failure_reason_by_stage.png", dpi=180)
        plt.close(fig)

    history_path = run_dir / "eval_history.jsonl"
    if history_path.exists():
        records = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        if records:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot([r["timesteps"] for r in records], [r["score"] for r in records], marker="o")
            ax.set_xlabel("timesteps")
            ax.set_ylabel("gated score")
            ax.set_title("Best Score Over Time")
            fig.tight_layout()
            fig.savefig(plots / "best_score_over_time.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot([r["timesteps"] for r in records], [r["highest_passed_stage"] for r in records], marker="o")
            ax.set_xlabel("timesteps")
            ax.set_ylabel("highest passed stage")
            ax.set_title("Curriculum Stage Over Time")
            fig.tight_layout()
            fig.savefig(plots / "curriculum_stage_over_time.png", dpi=180)
            plt.close(fig)

    print(f"Wrote workspace expansion plots to {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

