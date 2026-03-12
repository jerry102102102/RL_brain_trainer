"""WP1.5 rollout artifact schema and deterministic replay integrity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

ROLLOUT_ARTIFACT_SCHEMA_VERSION = "wp1.5.rollout.v1"


def validate_rollout_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a mapping"]
    if payload.get("schema_version") != ROLLOUT_ARTIFACT_SCHEMA_VERSION:
        errors.append(f"schema_version must be '{ROLLOUT_ARTIFACT_SCHEMA_VERSION}'")

    episode_id = payload.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        errors.append("episode_id must be a non-empty string")

    curriculum_level = payload.get("curriculum_level")
    if curriculum_level not in {"easy", "medium", "hard"}:
        errors.append("curriculum_level must be one of easy|medium|hard")

    seed = payload.get("seed")
    if not isinstance(seed, int):
        errors.append("seed must be an int")

    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("steps must be a non-empty list")
    else:
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"steps[{idx}] must be a mapping")
                continue
            if "t" not in step:
                errors.append(f"steps[{idx}].t is required")
            if "action" not in step:
                errors.append(f"steps[{idx}].action is required")
            if "reward_total" not in step:
                errors.append(f"steps[{idx}].reward_total is required")

    return errors


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def rollout_digest_sha256(payload: dict[str, Any]) -> str:
    errors = validate_rollout_payload(payload)
    if errors:
        raise ValueError("invalid rollout payload: " + "; ".join(errors))
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compare_replay_determinism(reference: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    ref_digest = rollout_digest_sha256(reference)
    replay_digest = rollout_digest_sha256(replay)
    match = ref_digest == replay_digest
    out = {
        "match": match,
        "reference_digest": ref_digest,
        "replay_digest": replay_digest,
        "schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
    }
    if not match:
        out["mismatch_reason"] = "canonical payload digest differs"
    return out
