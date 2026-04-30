"""Evaluate Finisher adaptation on Approach handoff states and clean resets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..training.policy_config import load_yaml_file, to_env_config, to_eval_config, write_json
from .eval_deterministic import _load_sb3_model
from .eval_dock import evaluate_dock_saved_model
from .eval_three_stage import _run_policy


def _mean(values: list[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_handoff_states(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    states = payload.get("states", payload if isinstance(payload, list) else [])
    if not isinstance(states, list):
        raise ValueError(f"Invalid handoff state buffer: {path}")
    return states


def _is_strict(*, pos: float, ori: float, env_cfg) -> bool:
    return bool(
        pos <= env_cfg.termination_config.success_pos_threshold_m
        and ori <= env_cfg.termination_config.success_ori_threshold_rad
    )


def _eval_model_on_handoff_states(
    *,
    checkpoint_path: Path,
    algorithm: str,
    env_cfg,
    states: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    model = _load_sb3_model(algorithm, checkpoint_path)
    metrics: list[dict[str, Any]] = []
    for idx, state in enumerate(states):
        entry_pos = float(state["position_error_norm"])
        entry_ori = float(state["orientation_error_norm"])
        entry_action = float(state.get("action_l2", 0.0))
        entry_dq = float(state.get("dq_norm", np.linalg.norm(state.get("initial_dq", [0.0] * 7))))
        env = ArmKinematicEnv(config=env_cfg)
        env.set_policy_mode("dock")
        result = _run_policy(
            env=env,
            model=model,
            reset_options={
                "initial_q": state["initial_q"],
                "initial_dq": state.get("initial_dq", [0.0] * 7),
                "initial_prev_action": state.get("initial_prev_action", [0.0] * 7),
                "goal_q": state["goal_q"],
                "goal_pose6": state["goal_pose6"],
                "policy_mode": "dock",
            },
        )
        final_pos = float(result["final_position_error"])
        final_ori = float(result["final_orientation_error"])
        final_action = float(result["final_action_magnitude"])
        final_dq = float(result["final_dq_norm"])
        entry_strict = _is_strict(pos=entry_pos, ori=entry_ori, env_cfg=env_cfg)
        final_strict = _is_strict(pos=final_pos, ori=final_ori, env_cfg=env_cfg)
        metrics.append(
            {
                "sample_id": idx,
                "success": bool(result["success"]),
                "entry_position_error": entry_pos,
                "entry_orientation_error": entry_ori,
                "entry_action_magnitude": entry_action,
                "entry_dq_norm": entry_dq,
                "final_position_error": final_pos,
                "final_orientation_error": final_ori,
                "final_action_magnitude": final_action,
                "final_dq_norm": final_dq,
                "entry_to_final_delta_position_error": final_pos - entry_pos,
                "entry_to_final_delta_orientation_error": final_ori - entry_ori,
                "entry_to_final_delta_action_magnitude": final_action - entry_action,
                "entry_to_final_delta_dq_norm": final_dq - entry_dq,
                "worse_than_entry": bool(final_pos > entry_pos + 1e-4 or final_ori > entry_ori + 1e-3),
                "entry_already_strict": bool(entry_strict),
                "strict_final": bool(final_strict),
                "strict_preserved": bool(entry_strict and final_strict),
                "strict_destroyed": bool(entry_strict and not final_strict),
            }
        )

    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "sample_count": len(metrics),
        "finisher_success_rate": _mean([m["success"] for m in metrics]),
        "strict_hold_rate": _mean([m["strict_final"] for m in metrics]),
        "worse_than_entry_rate": _mean([m["worse_than_entry"] for m in metrics]),
        "entry_already_strict_rate": _mean([m["entry_already_strict"] for m in metrics]),
        "strict_preserved_rate": _mean([m["strict_preserved"] for m in metrics]),
        "strict_destroyed_rate": _mean([m["strict_destroyed"] for m in metrics]),
        "mean_entry_position_error": _mean([m["entry_position_error"] for m in metrics]),
        "mean_entry_orientation_error": _mean([m["entry_orientation_error"] for m in metrics]),
        "mean_final_position_error": _mean([m["final_position_error"] for m in metrics]),
        "mean_final_orientation_error": _mean([m["final_orientation_error"] for m in metrics]),
        "mean_final_action_magnitude": _mean([m["final_action_magnitude"] for m in metrics]),
        "mean_final_dq_norm": _mean([m["final_dq_norm"] for m in metrics]),
        "entry_to_final_delta_position_error": _mean([m["entry_to_final_delta_position_error"] for m in metrics]),
        "entry_to_final_delta_orientation_error": _mean([m["entry_to_final_delta_orientation_error"] for m in metrics]),
        "entry_to_final_delta_action_magnitude": _mean([m["entry_to_final_delta_action_magnitude"] for m in metrics]),
        "entry_to_final_delta_dq_norm": _mean([m["entry_to_final_delta_dq_norm"] for m in metrics]),
        "episode_metrics": metrics,
    }


def _build_report(*, old_summary: dict[str, Any], adapted_summary: dict[str, Any], clean_summary: dict[str, Any]) -> str:
    old_success = float(old_summary["finisher_success_rate"])
    new_success = float(adapted_summary["finisher_success_rate"])
    conclusion = (
        "Adaptation improved Approach-handoff compatibility."
        if new_success > old_success
        else "Adaptation did not improve Approach-handoff compatibility yet."
    )
    return "\n".join(
        [
            "# Finisher Adaptation Report",
            "",
            "Purpose: test whether a hold-and-preserve Finisher is better at consuming the latest Approach handoff states.",
            "",
            "## Approach Handoff States",
            "",
            f"- Frozen old Finisher success: {old_summary['finisher_success_rate']:.3f}",
            f"- Adapted Finisher success: {adapted_summary['finisher_success_rate']:.3f}",
            f"- Frozen worse-than-entry rate: {old_summary['worse_than_entry_rate']:.3f}",
            f"- Adapted worse-than-entry rate: {adapted_summary['worse_than_entry_rate']:.3f}",
            f"- Adapted final action magnitude: {adapted_summary['mean_final_action_magnitude']:.6f}",
            f"- Adapted final dq norm: {adapted_summary['mean_final_dq_norm']:.6f}",
            "",
            "## Clean Reset Check",
            "",
            f"- Adapted clean strict success: {clean_summary.get('strict_success_rate', 0.0):.3f}",
            f"- Adapted clean mean final position error: {clean_summary.get('mean_final_position_error', 0.0):.6f}",
            f"- Adapted clean mean final orientation error: {clean_summary.get('mean_final_orientation_error', 0.0):.6f}",
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen vs adapted Finisher on Approach handoff states.")
    parser.add_argument("--old-finisher-checkpoint", required=True)
    parser.add_argument("--adapted-finisher-checkpoint", required=True)
    parser.add_argument("--finisher-config", required=True)
    parser.add_argument("--handoff-buffer", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--algorithm", default="td3")
    parser.add_argument("--episodes", type=int, default=100)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    env_cfg = to_env_config(load_yaml_file(Path(args.finisher_config)))
    eval_cfg = to_eval_config(load_yaml_file(Path(args.finisher_config)))
    states = _load_handoff_states(Path(args.handoff_buffer))[: max(int(args.episodes), 0)]
    old_summary = _eval_model_on_handoff_states(
        checkpoint_path=Path(args.old_finisher_checkpoint),
        algorithm=args.algorithm,
        env_cfg=env_cfg,
        states=states,
        label="frozen_old_finisher",
    )
    adapted_summary = _eval_model_on_handoff_states(
        checkpoint_path=Path(args.adapted_finisher_checkpoint),
        algorithm=args.algorithm,
        env_cfg=env_cfg,
        states=states,
        label="adapted_finisher",
    )
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    clean_summary = evaluate_dock_saved_model(
        algorithm=args.algorithm,
        checkpoint_path=Path(args.adapted_finisher_checkpoint),
        artifact_root=artifact_root / "clean_reset_eval",
        env_config=env_cfg,
        eval_config=eval_cfg,
    )
    combined = {
        "old_finisher_on_approach_handoff": {k: v for k, v in old_summary.items() if k != "episode_metrics"},
        "adapted_finisher_on_approach_handoff": {k: v for k, v in adapted_summary.items() if k != "episode_metrics"},
        "adapted_finisher_on_clean_reset": clean_summary,
    }
    write_json(artifact_root / "finisher_adaptation_eval_on_approach_handoff.json", {
        "old_finisher": old_summary,
        "adapted_finisher": adapted_summary,
    })
    write_json(artifact_root / "finisher_adaptation_eval_on_clean_reset.json", clean_summary)
    write_json(artifact_root / "finisher_adaptation_summary.json", combined)
    (artifact_root / "FINISHER_ADAPTATION_REPORT.md").write_text(
        _build_report(old_summary=old_summary, adapted_summary=adapted_summary, clean_summary=clean_summary)
    )
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
