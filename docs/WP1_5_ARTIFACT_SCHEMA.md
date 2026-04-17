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

## M2 RL action schema kickoff (`v2`)

`hrl_trainer.v5.rl_action` now accepts both `schema_version: v1` and `schema_version: v2`.

`v2` requirements (U-slot-first policy):
- Keep existing L2 fields: `skill_mode`, exactly one of `delta_pose|ee_target_pose`, `gripper_cmd`, `speed_profile_id`, `guard`.
- Primary control intent is `u_slot_params` + `timing_params` (not `gripper_cmd`).
- `gripper_cmd` is retained as a deprecated compatibility field in `v2`; accepted value is `HOLD` only.
- Explicitly rejected in `v2`: legacy gripper-first usage (`gripper_cmd=OPEN/CLOSE`, and legacy skill modes like `GRASP/LIFT/TRANSFER`).
- Add required `u_slot_params`:
  - `insert_depth` in `[0.0, 0.20]`
  - `lateral_alignment` in `[-0.10, 0.10]`
  - `vertical_clearance` in `[0.0, 0.20]`
  - `entry_yaw` in `[-pi, pi]`
- Add required `timing_params`:
  - `approach_speed_scale` in `[0.10, 2.00]`
  - `lift_profile_id` non-empty string
  - `contact_settle_time` in `[0.0, 2.0]`
- Optional guard hint: `fragility_mode_hint` in `{DEFAULT, CAUTIOUS, ROBUST}`.

Usage notes:
- Existing `validate_rl_action_v1(...)` is unchanged for legacy callers.
- New `validate_rl_action(...)` dispatches by `schema_version`.
- `action_to_skill_command(...)` is now version-aware and remains SkillCommand-only (no L3 trajectory fields).
