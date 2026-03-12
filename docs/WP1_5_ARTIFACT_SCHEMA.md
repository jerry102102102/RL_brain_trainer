# WP1.5 Artifact Schema and Versioning

Date: 2026-03-12
Scope: Phase-A scaffold for deterministic rollout/replay integrity and WP2 warm-start handoff.

## Version policy

- Each exported artifact must include a top-level `schema_version` string.
- Schema version changes follow `wp1.5.<artifact>.v<major>`.
- Backward-incompatible field changes require a new major version.

## Rollout artifact (`wp1.5.rollout.v1`)

Required top-level fields:
- `schema_version`: `wp1.5.rollout.v1`
- `episode_id`: stable identifier for one rollout
- `curriculum_level`: `easy|medium|hard`
- `seed`: deterministic episode seed
- `steps`: ordered step array

Minimum per-step fields:
- `t`: integer step index
- `action`: action payload used to drive L2 boundary
- `reward_total`: scalar step reward

Determinism rule:
- Canonical JSON (sorted keys, stable list order) is hashed with SHA-256.
- Replay integrity passes only if replay hash equals reference hash.

## Reward breakdown artifact (`wp1.5.reward_breakdown.v1`)

Required fields:
- `schema_version`
- `episode_id`
- `weights`: reward term weights
- `per_step`: per-term values per step
- `episode_total`

## Canonical trajectory artifact (`wp1.5.canonical_trajectory.v1`)

Required fields:
- `schema_version`
- `episode_id`
- `stage_sequence`
- `waypoints`
- `terminal_reason`

## CSV companion policy

If JSON artifacts are exported with CSV companions, CSV files must include:
- `schema_version`
- `episode_id`
- stable column order documented in exporter code

## Curriculum fixtures used in Phase-A scaffold

- `hrl_ws/src/hrl_trainer/config/v5_curriculum_easy.yaml`
- `hrl_ws/src/hrl_trainer/config/v5_curriculum_medium.yaml`
- `hrl_ws/src/hrl_trainer/config/v5_curriculum_hard.yaml`

These fixtures provide fixed seed sets for deterministic replay baselines.
