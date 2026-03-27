from __future__ import annotations

import pytest

from hrl_trainer.v5_1.contracts import SCHEMA_VERSION, validate_contract


def test_contract_observation_schema_passes() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "r1",
        "step_index": 0,
        "timestamp_ns": 123,
        "q": [0.0] * 6,
        "dq": [0.0] * 6,
        "ee_xyz": [0.0, 0.0, 0.0],
        "target_xyz": [0.1, 0.0, 0.2],
    }
    validate_contract("observation", payload)


def test_contract_rejects_missing_field() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "r1",
        "step_index": 0,
        "timestamp_ns": 123,
        "q": [0.0] * 6,
        "dq": [0.0] * 6,
        "ee_xyz": [0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError, match="missing required fields"):
        validate_contract("observation", payload)
