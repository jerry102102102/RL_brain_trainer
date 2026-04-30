# Phase 1C Bridge Policy Training Report

Purpose: This note records the first formal Phase 1C Bridge-policy implementation and training result. It is meant to separate the engineering result from the research result: the Bridge stack now trains and evaluates end-to-end, but the first learned Bridge policy does not yet create an overlap with the Dock acceptance basin.

## Objective

Phase 1C Bridge is introduced because the current two-policy pipeline has a distribution gap:

- Approach can bring the end-effector near the target in position.
- Dock is strong on its own clean local reset distribution.
- Real Approach handoff states and the Dock acceptance basin have almost no overlap.
- The dominant gap dimension measured so far is orientation, not coarse position.

The Bridge policy is therefore not a strict docking finisher. Its target role is to transform dirty near-goal handoff states into states that Dock can actually accept.

## Implemented Components

The implementation adds a trainable Bridge path while keeping the existing observation and action schema unchanged.

Implemented files:

- `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/bridge/bridge_reset_samplers.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/bridge/reward_bridge.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/bridge/train_bridge_policy.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/bridge/eval_bridge.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/bridge/switch_state_machine.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/bridge_default.yaml`
- `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/bridge_handoff_v1_12env.yaml`

The environment now supports `policy_mode = "bridge"` and sets `mode_flag = [0, 0, 1, 0]` for Bridge observations.

## Bridge Reset Distribution

Bridge reset states are sampled from the real Approach handoff dataset:

`artifacts/kinematic_phase1/phase1c/handoff_v1/handoff_dataset.jsonl`

The sampler filters for near-goal handoff candidates and assigns buckets:

- `dirty_orientation_bucket`: position is near, orientation is dirty.
- `dirty_motion_bucket`: position is near, motion/action magnitude is dirty.
- `mixed_dirty_bucket`: both orientation and motion are dirty.
- `near_goal_bucket`: fallback bucket for states that pass general filtering but are not strongly dirty.

The default Bridge training mix emphasizes dirty orientation states, because the acceptance-map comparison identified orientation as the primary gap.

## Bridge Success Definition

Bridge success is geometry-based basin entry, not strict docking:

- position error <= `0.008 m`
- orientation error <= `0.08 rad`

This is intentionally close to the Dock acceptance region measured by the acceptance map, while still being evaluated separately from Dock's final dwell-based strict success.

## Bridge Reward

The Bridge reward is designed around cleanup, not final docking:

- Orientation cleanup progress is the main positive term.
- Position progress/preservation prevents the Bridge from throwing away handoff proximity.
- Motion cleanup penalizes high joint motion.
- Acceptance-region bonus rewards entry into the Bridge success basin.
- Leave-near-goal, position regression, orientation regression, excessive action, and joint-limit penalties discourage unstable cleanup.

After the first smoke test, orientation progress was gated by position preservation. In the current version, orientation progress only counts when the state remains inside the Bridge work radius. This was added because the first trained policy found a bad shortcut: slightly improving orientation while leaving the near-goal region.

## Training Runs

### Smoke 001

Artifact:

`artifacts/kinematic_phase1/phase1c/bridge_handoff_v1_smoke_001`

Result:

- Bridge basin entry rate: `0.0`
- Bridge final position error: `0.0824 m`
- Bridge final orientation error: `2.1974 rad`
- Bridge leave-near-goal rate: `0.90`

Interpretation: the initial reward allowed the policy to sacrifice position to improve orientation. This was rejected as an invalid Bridge behavior.

### Smoke 002

Artifact:

`artifacts/kinematic_phase1/phase1c/bridge_handoff_v1_smoke_002`

Result:

- Bridge basin entry rate: `0.0`
- Bridge final position error: `0.0227 m`
- Bridge final orientation error: `2.2855 rad`
- Bridge leave-near-goal rate: `0.15`

Interpretation: stronger position preservation and lower action scale made the behavior safer, but did not yet solve the orientation cleanup problem.

### PPO 262k 001

Artifact:

`artifacts/kinematic_phase1/phase1c/bridge_handoff_v1_ppo_262k_001`

Result:

- Bridge basin entry rate: `0.0`
- Bridge final position error: `0.0786 m`
- Bridge final orientation error: `2.3006 rad`
- Bridge leave-near-goal rate: `0.95`
- Direct Dock success from the same handoff states: `0.0`
- Bridge -> Dock success: `0.0`

Interpretation: over longer training, the policy again exploited orientation progress at the cost of position. This confirmed the need to gate orientation reward on staying inside the Bridge work region.

### Position-Gated PPO 262k 001

Artifact:

`artifacts/kinematic_phase1/phase1c/bridge_handoff_v1_posgated_ppo_262k_001`

Result:

- Bridge basin entry rate: `0.0`
- Bridge final position error: `0.0663 m`
- Bridge final orientation error: `2.4521 rad`
- Bridge leave-near-goal rate: `0.83`
- Direct Dock success from the same handoff states: `0.0`
- Bridge -> Dock success: `0.0`

Interpretation: position-gated orientation reward reduced the exploit slightly but did not make Bridge successful. The current Bridge problem remains too hard for this first PPO + dense reward setup: it must reduce very large orientation error while preserving a near-goal position.

## Current Conclusion

Confirmed fact: the Phase 1C Bridge training and evaluation infrastructure is now functional. It can train a PPO Bridge policy from real handoff reset states and evaluate both Bridge-only basin entry and Bridge -> Dock completion against direct Dock.

Confirmed fact: the first Bridge policies do not yet create a usable overlap with Dock. Bridge -> Dock remains `0.0` success on the evaluated handoff states.

Current best interpretation: the remaining gap is not a switch-threshold problem. It is a hard local transformation problem: real Approach handoff states have very large orientation error, and a naive Bridge policy tends either to preserve position without fixing enough orientation or to improve orientation by leaving the near-goal region.

The immediate value of this step is diagnostic: Bridge is now a real trainable module, and the failure mode is measurable rather than guessed.
