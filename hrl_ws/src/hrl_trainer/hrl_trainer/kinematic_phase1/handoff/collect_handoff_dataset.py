"""Collect real approach rollout states that may be candidates for dock handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..eval.eval_deterministic import _load_sb3_model
from ..eval.fixed_eval_suite import build_fixed_eval_suite, suite_to_jsonable
from ..training.policy_config import (
    approach_default_config_path,
    deep_merge,
    load_yaml_file,
    to_env_config,
    write_json,
)
from .handoff_dataset import summarize_handoff_records, write_jsonl
from .handoff_features import annotate_handoff_record


def handoff_default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "handoff_dataset_default.yaml"


def _load_config(explicit_path: str | None) -> dict:
    cfg = load_yaml_file(handoff_default_config_path())
    if explicit_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(explicit_path)))
    return cfg


def _copy_obs(obs: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value, dtype=float).tolist() for key, value in obs.items()}


def _is_candidate(*, info: dict, handoff_cfg: dict) -> bool:
    pos = float(info["position_error_norm"])
    if pos <= float(handoff_cfg.get("candidate_pos_threshold_m", 0.03)):
        return True
    if bool(handoff_cfg.get("include_pre_near_goal", True)) and bool(info.get("curr_in_pre_near_goal", False)):
        return True
    if bool(handoff_cfg.get("include_near_goal", True)) and bool(info.get("curr_in_near_goal", False)):
        return True
    return False


def collect_handoff_dataset(
    *,
    approach_checkpoint: Path,
    approach_config_path: Path,
    artifact_root: Path,
    config: dict,
    approach_algorithm: str = "ppo",
) -> dict[str, object]:
    approach_cfg = deep_merge(load_yaml_file(approach_default_config_path()), load_yaml_file(approach_config_path))
    env_config = to_env_config(approach_cfg)
    handoff_cfg = config.get("handoff", {})
    model = _load_sb3_model(approach_algorithm, approach_checkpoint)
    suite = build_fixed_eval_suite(
        seed=int(handoff_cfg.get("suite_seed", 700001)),
        n_episodes=int(handoff_cfg.get("rollout_episodes", 200)),
        joint_specs=env_config.joint_specs,
        start_margin_fraction=env_config.start_sample_margin_fraction,
        goal_margin_fraction=env_config.goal_sample_margin_fraction,
    )
    max_samples = int(handoff_cfg.get("max_samples", 500))
    min_gap = int(handoff_cfg.get("min_steps_between_samples", 1))
    switch_rule = dict(handoff_cfg.get("switch_rule", {}))
    records: list[dict] = []
    env = ArmKinematicEnv(config=env_config)
    env.set_policy_mode("approach")
    for rollout_id, episode in enumerate(suite):
        obs, info = env.reset(options={**episode.reset_options(), "policy_mode": "approach"})
        terminated = truncated = False
        step = 0
        last_sample_step = -10_000
        prev_action_mag = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action, dtype=float)
            obs, _, terminated, truncated, info = env.step(action_arr)
            step += 1
            prev_action_mag = float(np.linalg.norm(action_arr))
            if step - last_sample_step < min_gap:
                continue
            if _is_candidate(info=info, handoff_cfg=handoff_cfg):
                records.append(
                    annotate_handoff_record(
                        observation=_copy_obs(obs),
                        info=info,
                        action_magnitude=prev_action_mag,
                        rollout_id=rollout_id,
                        episode_id=episode.episode_id,
                        step=step,
                        source_policy_checkpoint=str(approach_checkpoint),
                        switch_rule_config=switch_rule,
                    )
                )
                last_sample_step = step
                if len(records) >= max_samples:
                    break
        if len(records) >= max_samples:
            break

    dataset_path = artifact_root / "handoff_dataset.jsonl"
    summary = summarize_handoff_records(records)
    summary.update(
        {
            "dataset_path": str(dataset_path),
            "source_policy_checkpoint": str(approach_checkpoint),
            "approach_config_path": str(approach_config_path),
            "handoff_config": config,
            "suite_path": str(artifact_root / "handoff_collection_suite.json"),
        }
    )
    write_jsonl(dataset_path, records)
    write_json(artifact_root / "handoff_collection_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "handoff_dataset_summary.json", summary)
    return {"dataset_path": str(dataset_path), "summary_path": str(artifact_root / "handoff_dataset_summary.json"), "summary": summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Phase 1C approach handoff candidate states.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--approach-algorithm", default="ppo")
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    result = collect_handoff_dataset(
        approach_checkpoint=Path(args.approach_checkpoint),
        approach_config_path=Path(args.approach_config),
        artifact_root=Path(args.artifact_root),
        config=_load_config(args.config),
        approach_algorithm=args.approach_algorithm,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
