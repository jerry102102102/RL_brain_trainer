#!/usr/bin/env python3
"""Generate final report/PPT figures.

The script prefers official artifacts when available and falls back to the
fixed numbers in report/OFFICIAL_ARTIFACTS.md / final-package constants.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures"

STAGE_FALLBACK = [
    {"stage": 0, "success": 1.00, "handoff_pos_mm": 0.50, "handoff_ori": 0.0073, "final_pos_mm": 1.67, "final_ori": 0.0106},
    {"stage": 1, "success": 1.00, "handoff_pos_mm": 0.62, "handoff_ori": 0.0099, "final_pos_mm": 1.67, "final_ori": 0.0123},
    {"stage": 2, "success": 1.00, "handoff_pos_mm": 0.85, "handoff_ori": 0.0119, "final_pos_mm": 1.82, "final_ori": 0.0139},
    {"stage": 3, "success": 1.00, "handoff_pos_mm": 1.20, "handoff_ori": 0.0138, "final_pos_mm": 2.14, "final_ori": 0.0164},
    {"stage": 4, "success": 1.00, "handoff_pos_mm": 1.71, "handoff_ori": 0.0150, "final_pos_mm": 2.53, "final_ori": 0.0165},
    {"stage": 5, "success": 0.93, "handoff_pos_mm": 1.96, "handoff_ori": 0.0177, "final_pos_mm": 2.89, "final_ori": 0.0208},
]


def _load_json(path: Path):
    if not path.exists():
        print(f"WARN missing artifact, using fallback: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"WARN failed to read {path}: {exc}")
        return None


def _stage_rows() -> list[dict[str, float]]:
    payload = _load_json(ROOT / "artifacts/kinematic_phase1/phase1c/workspace_sweep_workspace_noop_vs_previous_summary_001.json")
    if not payload or "rows" not in payload:
        return STAGE_FALLBACK
    rows = []
    for row in payload["rows"]:
        new = row.get("new", {})
        rows.append(
            {
                "stage": int(row.get("stage_index", len(rows))),
                "success": float(new.get("success_rate", new.get("approach_to_finisher_success_rate", 0.0))),
                "handoff_pos_mm": float(new.get("mean_handoff_position_error", 0.0)) * 1000.0,
                "handoff_ori": float(new.get("mean_handoff_orientation_error", 0.0)),
                "final_pos_mm": float(new.get("mean_final_position_error", 0.0)) * 1000.0,
                "final_ori": float(new.get("mean_final_orientation_error", 0.0)),
            }
        )
    return rows or STAGE_FALLBACK


def architecture_diagram() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_axis_off()
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)
    colors = {
        "l1": "#2A9D8F",
        "l2": "#E76F51",
        "l3": "#264653",
        "soft": "#F4A261",
        "bg": "#F7F3EA",
    }
    fig.patch.set_facecolor(colors["bg"])
    boxes = [
        (0.45, 3.5, 2.0, 1.25, "User Command\nnatural language", "#ECE7D8"),
        (3.0, 3.5, 2.1, 1.25, "Qwen L1\nsemantic bridge", colors["l1"]),
        (5.65, 3.5, 2.15, 1.25, "IntentPacket\nvalidated task", "#55B7A8"),
        (8.35, 3.5, 2.4, 1.25, "L2 Skill Request\nAPPROACH -> FINISHER", colors["soft"]),
        (11.25, 3.5, 1.9, 1.25, "L3 Safety /\nExecution", colors["l3"]),
        (5.65, 1.35, 2.15, 1.15, "Route q_goal\nas observation", "#D7E7F7"),
        (8.35, 1.35, 2.4, 1.15, "RL Policy\njoint-delta action", "#F0C2B5"),
        (11.25, 1.35, 1.9, 1.15, "Robot / Gazebo\nvalidation", "#C9D7D3"),
    ]
    for x, y, w, h, label, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.2, edgecolor="#222", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12, weight="bold", color="white" if color in [colors["l1"], colors["l3"], colors["soft"]] else "#1A1A1A")
    arrow_pairs = [((2.45, 4.12), (3.0, 4.12)), ((5.1, 4.12), (5.65, 4.12)), ((7.8, 4.12), (8.35, 4.12)), ((10.75, 4.12), (11.25, 4.12)), ((7.8, 1.92), (8.35, 1.92)), ((10.75, 1.92), (11.25, 1.92)), ((9.55, 3.5), (9.55, 2.5))]
    for a, b in arrow_pairs:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=2.0, color="#222"))
    ax.text(3.95, 5.28, "L1 boundary: no raw action, no trajectory, no torque", ha="center", fontsize=11, color="#1f5f57")
    ax.text(9.85, 0.82, "L2/L3 owns policy rollout, safety, and execution", ha="center", fontsize=11, color="#783526")
    ax.set_title("Final Modular L1-to-RL Manipulation Stack", fontsize=18, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_architecture_l1_l2_l3.png", dpi=180)
    plt.close(fig)


def workspace_sweep() -> None:
    rows = _stage_rows()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    stages = [r["stage"] for r in rows]
    success = [r["success"] * 100 for r in rows]
    bars = ax.bar(stages, success, color=["#2A9D8F"] * 5 + ["#E76F51"])
    ax.set_ylim(0, 108)
    ax.set_xticks(stages)
    ax.set_xlabel("Workspace stage")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Approach -> Finisher Kinematic Workspace Sweep", weight="bold")
    for bar, val in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "workspace_sweep_stage_success.png", dpi=180)
    plt.close(fig)


def route_prefix_improvement() -> None:
    labels = ["Baseline\nfull route", "Prefix40", "Prefix80", "Prefix120", "Full483\nprobe"]
    values = [21, 40, 80, 120, 170]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    bars = ax.bar(labels, values, color=["#8D99AE", "#80B1D3", "#58A4B0", "#2A9D8F", "#E76F51"])
    ax.set_ylabel("Longest continuous successful prefix")
    ax.set_title("Route Curriculum Expands Sequential Route Coverage", weight="bold")
    ax.set_ylim(0, 190)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 4, str(val), ha="center", fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "route_prefix_improvement.png", dpi=180)
    plt.close(fig)


def route_limitations() -> None:
    labels = ["Stable\nvalidated", "Full probe\nreaches", "Route target", "Failed\nprefix180 latest"]
    values = [120, 170, 483, 1]
    colors = ["#2A9D8F", "#F4A261", "#264653", "#E76F51"]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Route waypoint index / prefix length")
    ax.set_title("Current Route Limitation: Full Transport Still Unsolved", weight="bold")
    ax.set_ylim(0, 520)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 10, str(val), ha="center", fontsize=11, weight="bold")
    ax.text(1.5, 430, "The system improved from local correction to long-prefix following,\nbut full holder1 -> holder8 remains future work.", ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "route_curriculum_limitations.png", dpi=180)
    plt.close(fig)


def main() -> None:
    architecture_diagram()
    workspace_sweep()
    route_prefix_improvement()
    route_limitations()
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
