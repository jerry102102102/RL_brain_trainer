"""Run full switched evaluation for separate approach and dock policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..envs.switching_wrapper import SwitchingConfig
from ..eval.eval_switched import evaluate_switched_policies
from .policy_config import (
    approach_default_config_path,
    deep_merge,
    dock_default_config_path,
    load_yaml_file,
    switch_default_config_path,
    to_env_config,
    to_eval_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate switched approach+dock policies.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--dock-checkpoint", required=True)
    parser.add_argument("--approach-config")
    parser.add_argument("--dock-config")
    parser.add_argument("--switch-config")
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--dock-algorithm", default="td3")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    parser = build_arg_parser()
    args = parser.parse_args()
    approach_cfg = load_yaml_file(approach_default_config_path())
    if args.approach_config:
        approach_cfg = deep_merge(approach_cfg, load_yaml_file(Path(args.approach_config)))
    dock_cfg = load_yaml_file(dock_default_config_path())
    if args.dock_config:
        dock_cfg = deep_merge(dock_cfg, load_yaml_file(Path(args.dock_config)))
    switch_cfg = load_yaml_file(switch_default_config_path())
    if args.switch_config:
        switch_cfg = deep_merge(switch_cfg, load_yaml_file(Path(args.switch_config)))

    approach_env_cfg = to_env_config(approach_cfg)
    dock_env_cfg = to_env_config(dock_cfg)
    eval_cfg = to_eval_config({"eval": deep_merge(approach_cfg.get("eval", {}), switch_cfg.get("eval", {}))})
    if args.seed is not None:
        eval_cfg = type(eval_cfg)(suite_seed=args.seed, episodes=eval_cfg.episodes, regression_tolerance_m=eval_cfg.regression_tolerance_m)
    if args.episodes is not None:
        eval_cfg = type(eval_cfg)(suite_seed=eval_cfg.suite_seed, episodes=args.episodes, regression_tolerance_m=eval_cfg.regression_tolerance_m)

    switch_settings = switch_cfg.get("switch", {})
    summary = evaluate_switched_policies(
        approach_algorithm=args.approach_algorithm,
        approach_checkpoint_path=Path(args.approach_checkpoint),
        dock_algorithm=args.dock_algorithm,
        dock_checkpoint_path=Path(args.dock_checkpoint),
        artifact_root=Path(args.artifact_root),
        approach_env_config=approach_env_cfg,
        dock_env_config=dock_env_cfg,
        eval_config=eval_cfg,
        switching_config=SwitchingConfig(**switch_settings),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
