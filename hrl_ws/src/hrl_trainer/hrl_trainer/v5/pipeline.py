"""V5 WP1 acceptance pipeline utilities for L1 intent generation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from .intent_layer import IntentResolutionError, SlotMap, build_intent_packet, load_runtime_slot_map


@dataclass(frozen=True)
class AcceptanceSummary:
    success_count: int
    fail_count: int
    fail_reason_breakdown: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "fail_reason_breakdown": dict(self.fail_reason_breakdown),
        }


def _slot_ids(slot_map: SlotMap) -> list[str]:
    return [slot.slot_id for slot in slot_map.slots]


def _default_object_estimates(slot_map: SlotMap, *, now_sec: float) -> list[dict[str, Any]]:
    object_ids = sorted({obj_id for slot in slot_map.slots for obj_id in slot.allowed_objects})
    return [
        {
            "object_id": object_id,
            "xyz": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "confidence": 0.99,
            "pos_std": 0.001,
            "yaw_std": 0.01,
            "stamp_sec": now_sec,
            "frame_id": "world",
        }
        for object_id in object_ids
    ]


def generate_smoke_commands(slot_map: SlotMap, count: int = 10) -> list[str]:
    ids = _slot_ids(slot_map)
    if len(ids) < 2:
        raise ValueError("Need at least 2 slots for acceptance tasks")
    commands: list[str] = []
    for idx in range(count):
        src = ids[idx % len(ids)]
        dst = ids[(idx + 1) % len(ids)]
        commands.append(f"MOVE_PLATE({src}, {dst})")
    return commands


def generate_random_commands(slot_map: SlotMap, count: int = 20, *, seed: int = 42) -> list[str]:
    ids = _slot_ids(slot_map)
    if len(ids) < 2:
        raise ValueError("Need at least 2 slots for acceptance tasks")
    rng = random.Random(seed)
    commands: list[str] = []
    for _ in range(count):
        src = rng.choice(ids)
        dst = rng.choice(ids)
        while dst == src:
            dst = rng.choice(ids)
        commands.append(f"MOVE_PLATE({src}, {dst})")
    return commands


def run_acceptance_commands(
    commands: Sequence[str],
    *,
    slot_map: SlotMap,
    object_estimates: Sequence[dict[str, Any]],
    now_sec: float,
) -> AcceptanceSummary:
    success_count = 0
    fail_reasons: Counter[str] = Counter()

    for command in commands:
        try:
            build_intent_packet(
                command,
                slot_map,
                object_estimates,
                now_sec=now_sec,
            )
            success_count += 1
        except IntentResolutionError as exc:
            fail_reasons[str(exc.code)] += 1
        except Exception as exc:  # pragma: no cover - defensive catch for harness accounting
            fail_reasons[type(exc).__name__] += 1

    fail_count = len(commands) - success_count
    return AcceptanceSummary(
        success_count=success_count,
        fail_count=fail_count,
        fail_reason_breakdown=dict(sorted(fail_reasons.items())),
    )


def run_wp1_acceptance(
    *,
    smoke_count: int = 10,
    random_count: int = 20,
    random_seed: int = 42,
    now_sec: float = 100.0,
    slot_map_path: str | None = None,
) -> dict[str, Any]:
    slot_map = load_runtime_slot_map(slot_map_path)
    object_estimates = _default_object_estimates(slot_map, now_sec=now_sec)

    smoke_commands = generate_smoke_commands(slot_map, smoke_count)
    random_commands = generate_random_commands(slot_map, random_count, seed=random_seed)

    smoke_summary = run_acceptance_commands(
        smoke_commands,
        slot_map=slot_map,
        object_estimates=object_estimates,
        now_sec=now_sec,
    )
    random_summary = run_acceptance_commands(
        random_commands,
        slot_map=slot_map,
        object_estimates=object_estimates,
        now_sec=now_sec,
    )

    fail_breakdown = Counter(smoke_summary.fail_reason_breakdown)
    fail_breakdown.update(random_summary.fail_reason_breakdown)

    return {
        "smoke": {
            "task_count": smoke_count,
            **smoke_summary.to_dict(),
        },
        "random": {
            "task_count": random_count,
            **random_summary.to_dict(),
        },
        "overall": {
            "task_count": smoke_count + random_count,
            "success_count": smoke_summary.success_count + random_summary.success_count,
            "fail_count": smoke_summary.fail_count + random_summary.fail_count,
            "fail_reason_breakdown": dict(sorted(fail_breakdown.items())),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V5 WP1 acceptance harness")
    parser.add_argument("--smoke-count", type=int, default=10)
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--now-sec", type=float, default=100.0)
    parser.add_argument("--slot-map", default=None, help="Override path to v5 slot map yaml")
    args = parser.parse_args()

    summary = run_wp1_acceptance(
        smoke_count=args.smoke_count,
        random_count=args.random_count,
        random_seed=args.seed,
        now_sec=args.now_sec,
        slot_map_path=args.slot_map,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
