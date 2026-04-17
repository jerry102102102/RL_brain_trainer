"""Deterministic Rule-L2 v0 baseline benchmark over synthetic episodes."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .intent_layer import SlotMap, build_intent_packet
from .rule_l2_v0 import RuleL2V0
from .trainer_loop import V5EpisodeSummary, run_v5_training_episode

BENCHMARK_SCHEMA = "v5_rule_l2_v0_benchmark"
BENCHMARK_VERSION = "1.0"


@dataclass(frozen=True)
class RuleL2V0BenchmarkSummary:
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


def _benchmark_slot_map() -> SlotMap:
    payload = {
        "slots": [
            {
                "slot_id": "shelf_A1",
                "region_world": {"center_xyz": [0.90, -1.16, 1.22], "size_xyz": [0.18, 0.18, 0.06], "yaw": 0.0},
                "approach_pose_candidates": [{"xyz": [0.86, -1.10, 1.32], "rpy": [3.14, 0.0, 0.0]}],
                "place_pose_candidates": [{"xyz": [0.90, -1.16, 1.22], "rpy": [3.14, 0.0, 0.0]}],
                "allowed_objects": ["tray1"],
                "priority": 1,
            },
            {
                "slot_id": "shelf_B1",
                "region_world": {"center_xyz": [-0.92, -1.16, 1.22], "size_xyz": [0.18, 0.18, 0.06], "yaw": 0.0},
                "approach_pose_candidates": [{"xyz": [-0.86, -1.10, 1.32], "rpy": [3.14, 0.0, 3.14]}],
                "place_pose_candidates": [{"xyz": [-0.92, -1.16, 1.22], "rpy": [3.14, 0.0, 3.14]}],
                "allowed_objects": ["tray1"],
                "priority": 1,
            },
        ]
    }
    return SlotMap.from_dict(payload)


def _object_estimates(now_sec: float) -> list[dict[str, Any]]:
    return [
        {
            "object_id": "tray1",
            "xyz": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "confidence": 0.99,
            "pos_std": 0.001,
            "yaw_std": 0.01,
            "stamp_sec": now_sec,
            "frame_id": "world",
        }
    ]


def _intent_command_for_episode(episode_index: int, rng: random.Random) -> str:
    if (episode_index + rng.randrange(2)) % 2 == 0:
        return "MOVE_PLATE(shelf_A1, shelf_B1)"
    return "MOVE_PLATE(shelf_B1, shelf_A1)"


def _step_inputs_for_rollout(rollout_length: int, terminal_success: bool, rng: random.Random) -> list[dict[str, Any]]:
    step_count = min(max(1, rollout_length), 3 + rng.randrange(3))
    rows: list[dict[str, Any]] = []
    for idx in range(step_count):
        terminal = idx == (step_count - 1)
        rows.append(
            {
                "potential_current": float(idx),
                "potential_next": float(idx + 1),
                "safety_violation": 0.0 if terminal_success else 0.05 * float((idx % 2) + 1),
                "action_delta": 0.05 * float((idx % 3) + 1),
                "coverage": 0.02 * float(idx + 1),
                "subgoal": 0.2 if terminal and terminal_success else 0.0,
                "terminal": terminal,
                "terminal_success": terminal_success,
            }
        )
    return rows


def _run_single_episode(
    *,
    episode_index: int,
    slot_map: SlotMap,
    policy: RuleL2V0,
    rng: random.Random,
) -> tuple[V5EpisodeSummary, bool]:
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
    terminal_success = rng.random() >= 0.35
    summary = run_v5_training_episode(
        episode_index=episode_index,
        step_inputs=_step_inputs_for_rollout(len(rollout), terminal_success, rng),
    )
    return summary, terminal_success


def run_rule_l2_v0_benchmark(*, episodes: int = 8, seed: int = 42) -> RuleL2V0BenchmarkSummary:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rng = random.Random(seed)
    slot_map = _benchmark_slot_map()
    policy = RuleL2V0()

    success_count = 0
    total_reward = 0.0
    total_episode_length = 0

    for episode_index in range(episodes):
        episode_summary, terminal_success = _run_single_episode(
            episode_index=episode_index,
            slot_map=slot_map,
            policy=policy,
            rng=rng,
        )
        success_count += 1 if terminal_success else 0
        total_reward += float(episode_summary.total_reward)
        total_episode_length += len(episode_summary.steps)

    fail_count = episodes - success_count
    return RuleL2V0BenchmarkSummary(
        schema=BENCHMARK_SCHEMA,
        version=BENCHMARK_VERSION,
        seed=int(seed),
        episode_count=int(episodes),
        success_count=int(success_count),
        fail_count=int(fail_count),
        average_reward=float(total_reward / episodes),
        average_episode_length=float(total_episode_length / episodes),
    )


def parse_rule_l2_v0_benchmark_summary(payload: Mapping[str, Any]) -> RuleL2V0BenchmarkSummary:
    required = {
        "schema",
        "version",
        "seed",
        "episode_count",
        "success_count",
        "fail_count",
        "average_reward",
        "average_episode_length",
    }
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Benchmark summary missing required fields: {missing}")

    summary = RuleL2V0BenchmarkSummary(
        schema=str(payload["schema"]),
        version=str(payload["version"]),
        seed=int(payload["seed"]),
        episode_count=int(payload["episode_count"]),
        success_count=int(payload["success_count"]),
        fail_count=int(payload["fail_count"]),
        average_reward=float(payload["average_reward"]),
        average_episode_length=float(payload["average_episode_length"]),
    )

    if summary.episode_count <= 0:
        raise ValueError("episode_count must be > 0")
    if summary.success_count < 0 or summary.fail_count < 0:
        raise ValueError("success_count/fail_count must be >= 0")
    if summary.success_count + summary.fail_count != summary.episode_count:
        raise ValueError("success_count + fail_count must equal episode_count")
    return summary


def load_rule_l2_v0_benchmark_summary(path: str | Path) -> RuleL2V0BenchmarkSummary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark summary payload must be a JSON object")
    return parse_rule_l2_v0_benchmark_summary(payload)


def write_rule_l2_v0_benchmark_summary(summary: RuleL2V0BenchmarkSummary, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    target.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Rule-L2 v0 benchmark")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/v5_rule_l2_v0_benchmark_summary.json")
    args = parser.parse_args(argv)

    summary = run_rule_l2_v0_benchmark(episodes=args.episodes, seed=args.seed)
    write_rule_l2_v0_benchmark_summary(summary, args.output)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
