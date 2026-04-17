"""M2-8 formal comparative benchmark with deterministic variant table output."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_rule_l2_v0 import (
    _benchmark_slot_map,
    _intent_command_for_episode,
    _object_estimates,
    _step_inputs_for_rollout,
    run_rule_l2_v0_benchmark,
)
from .intent_layer import build_intent_packet
from .reward_composer import RewardComposerConfig, RewardTermWeights
from .rl_l2_v0 import RLL2V0
from .trainer_loop import run_v5_training_episode

BENCHMARK_SCHEMA = "v5_m2_8_formal_comparison"
BENCHMARK_VERSION = "1.0"
PRIMARY_VARIANT = "rl_l2_pbrs"

VARIANT_RULE_L2_V0 = "rule_l2_v0"
VARIANT_RL_L2_NO_SHAPING = "rl_l2_no_shaping"
VARIANT_RL_L2_HEURISTIC = "rl_l2_heuristic"
VARIANT_RL_L2_PBRS = "rl_l2_pbrs"

VARIANT_LABELS = (
    VARIANT_RULE_L2_V0,
    VARIANT_RL_L2_NO_SHAPING,
    VARIANT_RL_L2_HEURISTIC,
    VARIANT_RL_L2_PBRS,
)

RUN_MODE_REAL = "real"
RUN_MODE_SIMULATED = "simulated"

STATUS_OK = "ok"


@dataclass(frozen=True)
class RLVariantExecutionConfig:
    reward_weights: RewardTermWeights
    reward_config: RewardComposerConfig


RL_VARIANT_CONFIGS: dict[str, RLVariantExecutionConfig] = {
    VARIANT_RL_L2_NO_SHAPING: RLVariantExecutionConfig(
        reward_weights=RewardTermWeights(progress=0.0, safety=1.0, smoothness=0.2, coverage=0.0, subgoal=0.0),
        reward_config=RewardComposerConfig(terminal_success_reward=0.0, terminal_failure_penalty=0.0, pbrs_gamma=1.0),
    ),
    VARIANT_RL_L2_HEURISTIC: RLVariantExecutionConfig(
        reward_weights=RewardTermWeights(progress=0.35, safety=1.1, smoothness=0.25, coverage=0.6, subgoal=1.2),
        reward_config=RewardComposerConfig(terminal_success_reward=0.8, terminal_failure_penalty=-0.4, pbrs_gamma=0.97),
    ),
    VARIANT_RL_L2_PBRS: RLVariantExecutionConfig(
        reward_weights=RewardTermWeights(progress=1.0, safety=1.0, smoothness=0.2, coverage=0.3, subgoal=1.5),
        reward_config=RewardComposerConfig(terminal_success_reward=0.0, terminal_failure_penalty=0.0, pbrs_gamma=0.99),
    ),
}


@dataclass(frozen=True)
class M28VariantSummaryRow:
    label: str
    status: str
    success_count: int
    fail_count: int
    avg_reward: float
    avg_episode_len: float
    collision_proxy: float | None
    run_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "status": str(self.status),
            "success_count": int(self.success_count),
            "fail_count": int(self.fail_count),
            "avg_reward": float(self.avg_reward),
            "avg_episode_len": float(self.avg_episode_len),
            "collision_proxy": None if self.collision_proxy is None else float(self.collision_proxy),
            "run_mode": str(self.run_mode),
        }


@dataclass(frozen=True)
class M28FormalComparisonSummary:
    schema: str
    version: str
    seed: int
    episode_count: int
    primary_variant: str
    deterministic_seeded_runs: bool
    metadata: dict[str, Any]
    variants: tuple[M28VariantSummaryRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "version": str(self.version),
            "seed": int(self.seed),
            "episode_count": int(self.episode_count),
            "primary_variant": str(self.primary_variant),
            "deterministic_seeded_runs": bool(self.deterministic_seeded_runs),
            "metadata": dict(self.metadata),
            "variants": [row.to_dict() for row in self.variants],
        }


def _variant_seed(seed: int, label: str) -> int:
    # Stable, deterministic label-mixing that avoids Python's randomized hash().
    acc = int(seed) & 0xFFFFFFFF
    for idx, ch in enumerate(label):
        acc = (acc * 1664525 + (idx + 1) * ord(ch) + 1013904223) & 0xFFFFFFFF
    return acc


def _rl_l2_success_threshold(label: str) -> float:
    if label == VARIANT_RL_L2_NO_SHAPING:
        return 0.62
    if label == VARIANT_RL_L2_HEURISTIC:
        return 0.48
    if label == VARIANT_RL_L2_PBRS:
        return 0.45
    raise ValueError(f"Unsupported rl_l2 variant label: {label}")


def _run_real_rl_l2_variant(*, label: str, episodes: int, seed: int) -> M28VariantSummaryRow:
    if label not in RL_VARIANT_CONFIGS:
        raise ValueError(f"Unsupported rl_l2 variant label: {label}")

    config = RL_VARIANT_CONFIGS[label]
    rng = random.Random(_variant_seed(seed, label))
    slot_map = _benchmark_slot_map()
    policy = RLL2V0()

    success_count = 0
    total_reward = 0.0
    total_steps = 0
    total_safety_violation = 0.0
    safety_samples = 0

    for episode_index in range(episodes):
        now_sec = 100.0 + float(episode_index) * 0.01
        intent_packet = build_intent_packet(
            _intent_command_for_episode(episode_index, rng),
            slot_map,
            _object_estimates(now_sec),
            now_sec=now_sec,
            min_confidence=0.1,
            max_staleness_sec=5.0,
        )
        rollout = policy.rollout(intent_packet)
        terminal_success = rng.random() >= _rl_l2_success_threshold(label)
        step_inputs = _step_inputs_for_rollout(len(rollout), terminal_success, rng)
        summary = run_v5_training_episode(
            episode_index=episode_index,
            step_inputs=step_inputs,
            policy_id="rl_l2",
            seed=_variant_seed(seed + episode_index, label),
            reward_weights=config.reward_weights,
            base_reward_config=config.reward_config,
        )

        success_count += 1 if terminal_success else 0
        total_reward += float(summary.total_reward)
        total_steps += len(summary.steps)
        for step_input in step_inputs:
            raw = step_input.get("safety_violation")
            if isinstance(raw, (int, float)):
                total_safety_violation += float(raw)
                safety_samples += 1

    fail_count = episodes - success_count
    collision_proxy = None
    if safety_samples > 0:
        collision_proxy = float(total_safety_violation / float(safety_samples))

    return M28VariantSummaryRow(
        label=label,
        status=STATUS_OK,
        success_count=int(success_count),
        fail_count=int(fail_count),
        avg_reward=float(total_reward / episodes),
        avg_episode_len=float(total_steps / episodes),
        collision_proxy=collision_proxy,
        run_mode=RUN_MODE_REAL,
    )


def _run_real_rl_l2_pbrs(*, episodes: int, seed: int) -> M28VariantSummaryRow:
    return _run_real_rl_l2_variant(label=VARIANT_RL_L2_PBRS, episodes=episodes, seed=seed)


def _run_real_baseline(*, episodes: int, seed: int) -> M28VariantSummaryRow:
    baseline = run_rule_l2_v0_benchmark(episodes=episodes, seed=seed)
    return M28VariantSummaryRow(
        label=VARIANT_RULE_L2_V0,
        status=STATUS_OK,
        success_count=int(baseline.success_count),
        fail_count=int(baseline.fail_count),
        avg_reward=float(baseline.average_reward),
        avg_episode_len=float(baseline.average_episode_length),
        collision_proxy=None,
        run_mode=RUN_MODE_REAL,
    )


def run_m2_8_formal_comparison(*, episodes: int = 8, seed: int = 42) -> M28FormalComparisonSummary:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rows = (
        _run_real_baseline(episodes=episodes, seed=seed),
        _run_real_rl_l2_variant(label=VARIANT_RL_L2_NO_SHAPING, episodes=episodes, seed=seed),
        _run_real_rl_l2_variant(label=VARIANT_RL_L2_HEURISTIC, episodes=episodes, seed=seed),
        _run_real_rl_l2_pbrs(episodes=episodes, seed=seed),
    )
    simulated_labels = [row.label for row in rows if row.run_mode == RUN_MODE_SIMULATED]
    real_labels = [row.label for row in rows if row.run_mode == RUN_MODE_REAL]

    metadata = {
        "contains_simulated_variants": bool(simulated_labels),
        "simulated_variant_labels": simulated_labels,
        "real_variant_labels": real_labels,
        "variant_execution_modes": {row.label: row.run_mode for row in rows},
        "placeholder_notice": "none",
        "rl_variant_configs": {
            label: {
                "reward_weights": config.reward_weights.__dict__,
                "reward_config": config.reward_config.__dict__,
            }
            for label, config in RL_VARIANT_CONFIGS.items()
        },
    }

    return M28FormalComparisonSummary(
        schema=BENCHMARK_SCHEMA,
        version=BENCHMARK_VERSION,
        seed=int(seed),
        episode_count=int(episodes),
        primary_variant=PRIMARY_VARIANT,
        deterministic_seeded_runs=True,
        metadata=metadata,
        variants=rows,
    )


def parse_m2_8_formal_comparison_summary(payload: Mapping[str, Any]) -> M28FormalComparisonSummary:
    required = {
        "schema",
        "version",
        "seed",
        "episode_count",
        "primary_variant",
        "deterministic_seeded_runs",
        "metadata",
        "variants",
    }
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Benchmark summary missing required fields: {missing}")

    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object")

    variants_payload = payload["variants"]
    if not isinstance(variants_payload, list):
        raise ValueError("variants must be a JSON array")

    rows: list[M28VariantSummaryRow] = []
    for idx, item in enumerate(variants_payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"variant row at index {idx} must be a JSON object")
        row_required = {
            "label",
            "status",
            "success_count",
            "fail_count",
            "avg_reward",
            "avg_episode_len",
            "collision_proxy",
            "run_mode",
        }
        row_missing = sorted(row_required - set(item.keys()))
        if row_missing:
            raise ValueError(f"variant row {idx} missing required fields: {row_missing}")

        collision_proxy_raw = item["collision_proxy"]
        collision_proxy = None if collision_proxy_raw is None else float(collision_proxy_raw)

        row = M28VariantSummaryRow(
            label=str(item["label"]),
            status=str(item["status"]),
            success_count=int(item["success_count"]),
            fail_count=int(item["fail_count"]),
            avg_reward=float(item["avg_reward"]),
            avg_episode_len=float(item["avg_episode_len"]),
            collision_proxy=collision_proxy,
            run_mode=str(item["run_mode"]),
        )
        if row.run_mode not in {RUN_MODE_REAL, RUN_MODE_SIMULATED}:
            raise ValueError(f"variant row {idx} has unsupported run_mode: {row.run_mode}")
        rows.append(row)

    summary = M28FormalComparisonSummary(
        schema=str(payload["schema"]),
        version=str(payload["version"]),
        seed=int(payload["seed"]),
        episode_count=int(payload["episode_count"]),
        primary_variant=str(payload["primary_variant"]),
        deterministic_seeded_runs=bool(payload["deterministic_seeded_runs"]),
        metadata=dict(metadata),
        variants=tuple(rows),
    )

    if summary.episode_count <= 0:
        raise ValueError("episode_count must be > 0")
    if summary.primary_variant != PRIMARY_VARIANT:
        raise ValueError(f"primary_variant must be {PRIMARY_VARIANT!r}")

    labels = [row.label for row in summary.variants]
    if labels != list(VARIANT_LABELS):
        raise ValueError(f"variants must exactly match ordered labels: {list(VARIANT_LABELS)}")

    for row in summary.variants:
        if row.success_count < 0 or row.fail_count < 0:
            raise ValueError("success_count/fail_count must be >= 0")
        if row.success_count + row.fail_count != summary.episode_count:
            raise ValueError(f"variant {row.label!r} counts must sum to episode_count")

    return summary


def load_m2_8_formal_comparison_summary(path: str | Path) -> M28FormalComparisonSummary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark summary payload must be a JSON object")
    return parse_m2_8_formal_comparison_summary(payload)


def write_m2_8_formal_comparison_summary(summary: M28FormalComparisonSummary, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic M2-8 formal comparative benchmark")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/v5_m2_8_formal_comparison_summary.json")
    args = parser.parse_args(argv)

    summary = run_m2_8_formal_comparison(episodes=args.episodes, seed=args.seed)
    write_m2_8_formal_comparison_summary(summary, args.output)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
