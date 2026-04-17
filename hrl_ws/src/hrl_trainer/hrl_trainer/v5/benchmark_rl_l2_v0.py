"""Minimal RL-L2 v0 benchmark over synthetic episodes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .benchmark_rule_l2_v0 import (
    _benchmark_slot_map,
    _intent_command_for_episode,
    _object_estimates,
    _step_inputs_for_rollout,
)
from .intent_layer import build_intent_packet
from .rl_l2_v0 import RLL2V0
from .trainer_loop import V5EpisodeSummary, run_v5_training_episode

BENCHMARK_SCHEMA = "v5_rl_l2_v0_benchmark"
BENCHMARK_VERSION = "1.0"


@dataclass(frozen=True)
class RLL2V0BenchmarkSummary:
    schema: str
    version: str
    seed: int
    episode_count: int
    success_count: int
    fail_count: int
    average_reward: float
    average_episode_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "seed": int(self.seed),
            "episode_count": int(self.episode_count),
            "success_count": int(self.success_count),
            "fail_count": int(self.fail_count),
            "average_reward": float(self.average_reward),
            "average_episode_length": float(self.average_episode_length),
        }


def _run_single_episode(*, episode_index: int, policy: RLL2V0, rng: random.Random) -> tuple[V5EpisodeSummary, bool]:
    slot_map = _benchmark_slot_map()
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
    terminal_success = rng.random() >= 0.45
    summary = run_v5_training_episode(
        episode_index=episode_index,
        step_inputs=_step_inputs_for_rollout(len(rollout), terminal_success, rng),
    )
    return summary, terminal_success


def run_rl_l2_v0_benchmark(*, episodes: int = 8, seed: int = 42) -> RLL2V0BenchmarkSummary:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rng = random.Random(seed)
    policy = RLL2V0()
    success_count = 0
    total_reward = 0.0
    total_episode_length = 0

    for episode_index in range(episodes):
        episode_summary, terminal_success = _run_single_episode(episode_index=episode_index, policy=policy, rng=rng)
        success_count += 1 if terminal_success else 0
        total_reward += float(episode_summary.total_reward)
        total_episode_length += len(episode_summary.steps)

    fail_count = episodes - success_count
    return RLL2V0BenchmarkSummary(
        schema=BENCHMARK_SCHEMA,
        version=BENCHMARK_VERSION,
        seed=int(seed),
        episode_count=int(episodes),
        success_count=int(success_count),
        fail_count=int(fail_count),
        average_reward=float(total_reward / episodes),
        average_episode_length=float(total_episode_length / episodes),
    )
