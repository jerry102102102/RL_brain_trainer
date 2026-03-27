"""Minimal V5.1 L1/L2/L3 smoke pipeline with layer logs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .contracts import (
    SCHEMA_VERSION,
    ActionCommand,
    LayerLogRecord,
    ObservationFrame,
    to_payload,
    validate_contract,
)
from .l3_executor import L3DeterministicExecutor, L3ExecutorConfig
from .safety_watchdog import Intervention, SafetyWatchdog


def _jsonl_append(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def run_smoke(run_id: str, steps: int, log_root: Path) -> dict[str, str]:
    ts0 = time.time_ns()
    q = np.zeros(6, dtype=float)
    dq = np.zeros(6, dtype=float)
    target_q = np.array([0.2, -0.15, 0.1, 0.05, 0.0, 0.0], dtype=float)

    executor = L3DeterministicExecutor(L3ExecutorConfig(dt=0.1))
    watchdog = SafetyWatchdog(timeout_s=0.35, timeout_action=Intervention.HOLD)

    l1_path = log_root / "l1" / f"{run_id}.jsonl"
    l2_path = log_root / "l2" / f"{run_id}.jsonl"
    l3_path = log_root / "l3" / f"{run_id}.jsonl"

    prev_q_des: np.ndarray | None = None
    now_s = 0.0

    for step in range(max(1, int(steps))):
        now_ns = ts0 + step * 100_000_000
        now_s = step * 0.1

        obs = ObservationFrame(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            step_index=step,
            timestamp_ns=now_ns,
            q=q.tolist(),
            dq=dq.tolist(),
            ee_xyz=q[:3].tolist(),
            target_xyz=target_q[:3].tolist(),
        )
        obs_payload = to_payload(obs)
        validate_contract("observation", obs_payload)
        _jsonl_append(
            l1_path,
            to_payload(
                LayerLogRecord(
                    schema_version=SCHEMA_VERSION,
                    run_id=run_id,
                    layer="L1",
                    step_index=step,
                    timestamp_ns=now_ns,
                    payload=obs_payload,
                )
            ),
        )

        delta_q = (target_q - q) * 0.5
        action = ActionCommand(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            step_index=step,
            timestamp_ns=now_ns,
            source="l2_policy",
            delta_q=delta_q.tolist(),
        )
        action_payload = to_payload(action)
        validate_contract("action", action_payload)
        _jsonl_append(
            l2_path,
            to_payload(
                LayerLogRecord(
                    schema_version=SCHEMA_VERSION,
                    run_id=run_id,
                    layer="L2",
                    step_index=step,
                    timestamp_ns=now_ns,
                    payload=action_payload,
                )
            ),
        )

        result = executor.compute_q_des(q_current=q, delta_q_cmd=delta_q, prev_q_des=prev_q_des)
        watchdog.observe_command(now_s=now_s, q_current=result.q_des)
        wd = watchdog.evaluate(now_s=now_s, q_current=result.q_des)

        q_next = wd.q_command if wd.q_command is not None else result.q_des
        dq = (q_next - q) / 0.1
        q = q_next
        prev_q_des = result.q_des

        l3_payload = {
            "requested_delta_q": result.requested_delta_q.tolist(),
            "clamped_delta_q": result.clamped_delta_q.tolist(),
            "limited_q_des": result.limited_q_des.tolist(),
            "q_des": result.q_des.tolist(),
            "projection_applied": result.projection_applied,
            "watchdog": {
                "intervention": wd.intervention.value,
                "reason": wd.reason,
            },
        }
        _jsonl_append(
            l3_path,
            to_payload(
                LayerLogRecord(
                    schema_version=SCHEMA_VERSION,
                    run_id=run_id,
                    layer="L3",
                    step_index=step,
                    timestamp_ns=now_ns,
                    payload=l3_payload,
                )
            ),
        )

    return {"l1": str(l1_path), "l2": str(l2_path), "l3": str(l3_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal V5.1 L1/L2/L3 smoke pipeline")
    parser.add_argument("--run-id", default=f"smoke_{int(time.time())}")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--log-root", default="artifacts/v5_1/logs")
    args = parser.parse_args()

    outputs = run_smoke(run_id=args.run_id, steps=args.steps, log_root=Path(args.log_root))
    print(json.dumps({"run_id": args.run_id, "logs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
