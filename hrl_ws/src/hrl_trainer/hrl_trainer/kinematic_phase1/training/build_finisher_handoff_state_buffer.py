"""Build final-settled Approach handoff states for Finisher adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..eval.eval_deterministic import _load_sb3_model
from ..eval.eval_pipeline_ablation import _run_approach_with_handoff
from ..eval.fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable
from .policy_config import approach_default_config_path, deep_merge, load_yaml_file, to_env_config, write_json


def _finisher_ready(result: dict[str, Any], *, cfg) -> bool:
    return bool(
        cfg.finisher_ready_pos_threshold_m > 0.0
        and cfg.finisher_ready_ori_threshold_rad > 0.0
        and float(result["final_position_error"]) <= cfg.finisher_ready_pos_threshold_m
        and float(result["final_orientation_error"]) <= cfg.finisher_ready_ori_threshold_rad
        and (cfg.finisher_ready_action_threshold <= 0.0 or float(result["final_action_magnitude"]) <= cfg.finisher_ready_action_threshold)
        and (cfg.finisher_ready_dq_threshold <= 0.0 or float(result["final_dq_norm"]) <= cfg.finisher_ready_dq_threshold)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final-settled handoff states from an Approach policy.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--handoff-confirm-steps", type=int, default=2)
    parser.add_argument("--handoff-mode", choices=("final_settled", "first_confirmed", "final_always"), default="final_settled")
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    cfg = deep_merge(load_yaml_file(approach_default_config_path()), load_yaml_file(Path(args.approach_config)))
    env_cfg = to_env_config(cfg)
    model = _load_sb3_model(args.approach_algorithm, Path(args.approach_checkpoint))
    if env_cfg.curriculum_config.enabled and env_cfg.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(
            env_cfg,
            seed=args.seed,
            stage_index=args.stage_index,
            n_episodes=args.episodes,
        )
        eval_scope = "curriculum_region"
    else:
        suite = build_fixed_eval_suite(
            seed=args.seed,
            n_episodes=args.episodes,
            joint_specs=env_cfg.joint_specs,
            start_margin_fraction=env_cfg.start_sample_margin_fraction,
            goal_margin_fraction=env_cfg.goal_sample_margin_fraction,
        )
        eval_scope = "fixed_random"

    states: list[dict[str, object]] = []
    episode_summaries: list[dict[str, object]] = []
    for episode in suite:
        env = ArmKinematicEnv(config=env_cfg)
        env.set_curriculum_stage(args.stage_index)
        env.set_policy_mode("approach")
        approach_result, first_handoff = _run_approach_with_handoff(
            env=env,
            model=model,
            reset_options={**episode.reset_options(), "policy_mode": "approach"},
            ready_cfg=env_cfg.reward_config,
            handoff_confirm_steps=args.handoff_confirm_steps,
        )
        final_ready = _finisher_ready(approach_result, cfg=env_cfg.reward_config)
        if args.handoff_mode == "final_settled":
            handoff = approach_result if final_ready else None
        elif args.handoff_mode == "first_confirmed":
            handoff = first_handoff
        else:
            handoff = approach_result

        if handoff is not None:
            states.append(
                {
                    "episode_id": int(episode.episode_id),
                    "step_index": int(handoff["step_count"]),
                    "initial_q": handoff["final_q"],
                    "initial_dq": handoff["final_dq"],
                    "initial_prev_action": handoff["final_prev_action"],
                    "goal_q": handoff["goal_q"],
                    "goal_pose6": handoff["goal_pose6"],
                    "position_error_norm": float(handoff["final_position_error"]),
                    "orientation_error_norm": float(handoff["final_orientation_error"]),
                    "dwell_count": int(env_cfg.dwell_steps_target),
                    "action_l2": float(handoff["final_action_magnitude"]),
                    "dq_norm": float(handoff["final_dq_norm"]),
                    "source_checkpoint_name": str(Path(args.approach_checkpoint).name),
                    "handoff_mode": args.handoff_mode,
                }
            )
        episode_summaries.append(
            {
                "episode_id": int(episode.episode_id),
                "stored_handoff": bool(handoff is not None),
                "final_ready": bool(final_ready),
                "final_position_error": float(approach_result["final_position_error"]),
                "final_orientation_error": float(approach_result["final_orientation_error"]),
                "final_action_magnitude": float(approach_result["final_action_magnitude"]),
                "final_dq_norm": float(approach_result["final_dq_norm"]),
            }
        )

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    pos_errors = [float(item["position_error_norm"]) for item in states]
    ori_errors = [float(item["orientation_error_norm"]) for item in states]
    action_l2 = [float(item["action_l2"]) for item in states]
    dq_norms = [float(item["dq_norm"]) for item in states]
    summary = {
        "source_approach_checkpoint": str(Path(args.approach_checkpoint)),
        "source_approach_config": str(Path(args.approach_config)),
        "approach_algorithm": args.approach_algorithm,
        "handoff_mode": args.handoff_mode,
        "eval_scope": eval_scope,
        "episode_count": len(suite),
        "stored_handoff_count": len(states),
        "stored_handoff_rate": float(len(states) / len(suite)) if suite else 0.0,
        "mean_position_error": float(np.mean(pos_errors)) if pos_errors else None,
        "mean_orientation_error": float(np.mean(ori_errors)) if ori_errors else None,
        "mean_action_l2": float(np.mean(action_l2)) if action_l2 else None,
        "mean_dq_norm": float(np.mean(dq_norms)) if dq_norms else None,
        "states": states,
        "episode_summaries": episode_summaries,
    }
    write_json(artifact_root / "finisher_handoff_state_buffer_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "finisher_handoff_state_buffer.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"states", "episode_summaries"}}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
