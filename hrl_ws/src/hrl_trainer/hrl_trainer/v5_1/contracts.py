"""Frozen V5.1 contracts for L1/L2/L3 interfaces.

These schemas are intentionally minimal for T1 and stable enough for smoke validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ObservationFrame:
    schema_version: str
    run_id: str
    step_index: int
    timestamp_ns: int
    q: list[float]
    dq: list[float]
    ee_xyz: list[float]
    target_xyz: list[float]


@dataclass(frozen=True)
class ActionCommand:
    schema_version: str
    run_id: str
    step_index: int
    timestamp_ns: int
    source: Literal["l2_policy", "watchdog_hold", "watchdog_stop"]
    delta_q: list[float]


@dataclass(frozen=True)
class LayerLogRecord:
    schema_version: str
    run_id: str
    layer: Literal["L1", "L2", "L3"]
    step_index: int
    timestamp_ns: int
    payload: dict[str, Any]


SCHEMA_VERSION = "v5_1.contracts.v1"

SCHEMAS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "observation": {
        "schema_version": str,
        "run_id": str,
        "step_index": int,
        "timestamp_ns": int,
        "q": list,
        "dq": list,
        "ee_xyz": list,
        "target_xyz": list,
    },
    "action": {
        "schema_version": str,
        "run_id": str,
        "step_index": int,
        "timestamp_ns": int,
        "source": str,
        "delta_q": list,
    },
    "layer_log": {
        "schema_version": str,
        "run_id": str,
        "layer": str,
        "step_index": int,
        "timestamp_ns": int,
        "payload": dict,
    },
}


def validate_contract(kind: Literal["observation", "action", "layer_log"], payload: dict[str, Any]) -> None:
    """Strict structural validator for contract freeze checks."""
    schema = SCHEMAS[kind]
    missing = [key for key in schema if key not in payload]
    if missing:
        raise ValueError(f"{kind}: missing required fields: {missing}")

    extra = [key for key in payload if key not in schema]
    if extra:
        raise ValueError(f"{kind}: unexpected fields: {extra}")

    for key, expected_type in schema.items():
        value = payload[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{kind}.{key}: expected {expected_type}, got {type(value)}"
            )

    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{kind}.schema_version mismatch: {payload['schema_version']} != {SCHEMA_VERSION}"
        )


def to_payload(record: ObservationFrame | ActionCommand | LayerLogRecord) -> dict[str, Any]:
    return asdict(record)
