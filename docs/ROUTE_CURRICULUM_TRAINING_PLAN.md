# Route Curriculum Training Plan

Purpose: This document defines the next training direction for turning the current local RL precision controller into a scene-level route-following controller. Gazebo is reserved for final validation; the main training and diagnosis loop should first run in the pure kinematic environment.

## 1. Background

The current RL policy is not useless. It is a local precision controller.

Confirmed results:

```text
Early dense q-goal waypoint probe in Gazebo:
handoff confirmed rate = 1.0
mean final position error ~= 0.00234 m
```

But full-route validation fails:

```text
Full dense route numeric validation:
target count = 483
success rate = 0.0435
longest continuous success prefix = 21 waypoints
successful prefix distance ~= 0.0165 m
full route distance ~= 5.84 m
```

The current policy has learned local waypoint correction, not full holder1-to-holder8 transport.

## 2. Goal

The goal is to train the RL policy to follow the scene-level dense route using route curriculum training.

The policy must still output normalized joint-delta actions. The route q-goal is a target observation, not a controller command.

The target progression is:

```text
local correction
-> stable route prefix following
-> route segment recovery
-> full dense route following
-> Gazebo validation
```

## 3. Non-Goals

This phase does not:

- replace RL with IK or a classical controller
- use IK as the runtime controller
- reopen Dock-Coarse or Bridge as the main path
- change the core observation/action contract
- train in Gazebo
- touch real hardware
- require image-grounded perception

## 4. Route Dataset

The route dataset should be loaded from the existing dense q-goal route artifact.

Each route waypoint should expose:

- `route_index`
- `q_goal`
- `ee_target_position`
- `ee_target_orientation`
- `next_q_delta`
- `route_progress_m`
- `chunk_id`

The main route artifact is:

```text
artifacts/kinematic_phase1/phase1c/scene_route_curriculum/tray1_holder1_to_8_route_q_dense.json
```

The route dataset should compute FK target pose and route progress from q-goal values.

## 5. Route Reset Modes

The route training system should support several reset modes:

- `prefix_start_reset`: start from the route beginning.
- `random_prefix_reset`: sample starts within the unlocked prefix.
- `segment_reset`: sample starts inside a fixed route chunk.
- `recovery_reset`: reset near route waypoints with q/dq/prev-action noise.

The important change is that training must not only start near route index 0. It must learn to recover from the middle of the route.

## 6. Curriculum

The curriculum should be prefix- and segment-based, not only stride-based.

Prefix stages:

```text
prefix_20
prefix_40
prefix_80
prefix_120
prefix_180
prefix_260
prefix_360
prefix_483
```

Route chunks:

```text
chunk_0: 1-40
chunk_1: 41-80
chunk_2: 81-120
chunk_3: 121-180
chunk_4: 181-260
chunk_5: 261-360
chunk_6: 361-483
```

Promotion should require more than reward:

- segment success rate
- same-step route-ready hit rate
- orientation gate hit rate
- regression rate

Suggested promotion thresholds:

```text
promotion_success_rate >= 0.75
promotion_route_ready_hit_rate >= 0.75
promotion_orientation_hit_rate >= 0.75
promotion_regression_rate <= 0.25
promotion_window_episodes = 256
```

## 7. Route Reward

Route reward should be explicit and route-specific.

Dense reward terms:

- `q_goal_progress`
- `ee_position_progress`
- `ee_orientation_progress`
- `route_tangent_progress_bonus`
- `same_step_route_ready_bonus`
- `route_ready_dwell_bonus`
- `low_motion_near_waypoint_bonus`

Penalties:

- `orientation_regression_penalty`
- `q_route_regression_penalty`
- `off_route_penalty`
- `action_smoothness_penalty`
- `dq_penalty`
- `no_progress_penalty`

Initial waypoint success can be:

```text
position_error <= 0.01 m
orientation_error <= 0.15 rad
q_error <= configured q threshold
dwell >= 1 or 2
```

The first known full-route failure is orientation-driven, so orientation continuity must be a first-class signal.

## 8. Evaluators

Three evaluators are required:

### Teacher-forced waypoint eval

Each waypoint starts near its reference q-goal. This measures local route target ability.

### Sequential actual-final-q eval

This is the main evaluator.

```text
start at route beginning
target waypoint i
roll policy to final_q
use actual final_q as start for waypoint i+1
continue through prefix or full route
```

This measures whether errors accumulate or the policy can keep following the route.

### Recovery eval

Start from route segments with q/dq/action noise and measure recovery success by chunk.

## 9. Main KPIs

The main KPIs are:

- `longest_success_prefix`
- `cumulative_successful_route_distance_m`
- `chunk_0_success_rate`
- `chunk_1_success_rate`
- `first_failure_index`
- `first_failure_reason`
- `sequential_full_route_success_rate`
- `mean_final_orientation_error_by_chunk`
- `recovery_success_rate_by_segment`

If local waypoint success improves but sequential prefix does not grow, the training is not solving the route transport problem.

## 10. Acceptance Targets

Minimum target:

```text
longest_success_prefix > 40
```

Medium target:

```text
chunk_1 (indices 41-80) no longer has 0 success
```

High target:

```text
prefix_80 or prefix_120 sequential eval is stable
```

Highest target:

```text
full 483 route success rate clearly above 0.0435
mean final error no longer explodes
```

## 11. Gazebo Integration

Gazebo should only be used after numeric route evaluation improves.

The route-trained checkpoint should export:

- model checkpoint
- route config
- route evaluator output
- route rollout trace
- optional Phase 3A route-following request format

## 12. One-Sentence Summary

Route Curriculum Training should expand the current local waypoint correction policy into a scene-level route-following RL controller through explicit prefix curriculum, segment reset, route-specific reward, and sequential actual-final-q evaluation.

## 13. Implementation Status

The first Route Curriculum Training implementation is now present in the repo.

Core modules:

- `hrl_trainer.kinematic_phase1.route.route_dataset`: loads dense q-goal routes and computes FK target pose, route tangent, chunk id, and cumulative progress.
- `hrl_trainer.kinematic_phase1.route.route_reset_samplers`: supports prefix, random-prefix, segment, and recovery reset modes.
- `hrl_trainer.kinematic_phase1.route.reward_route`: implements route-specific q-goal, EE position, EE orientation, tangent, readiness, dwell, smoothness, and regression reward terms.
- `hrl_trainer.kinematic_phase1.route.route_env`: single-waypoint route target environment.
- `hrl_trainer.kinematic_phase1.route.route_sequence_env`: sequential mini-route environment that advances from one route target to the next using the actual final q.
- `hrl_trainer.kinematic_phase1.route.route_observation`: adds route-specific observation keys: `route_q_goal`, `route_q_error`, `route_tangent`, and `route_scalar`.
- `hrl_trainer.kinematic_phase1.eval.eval_route_curriculum`: sequential actual-final-q evaluator with chunk metrics and first-failure reporting.
- `hrl_trainer.kinematic_phase1.train_route_curriculum`: PPO training entry for route curriculum runs.

Important design correction:

```text
The first route curriculum attempts had q_goal_progress in the reward,
but the policy did not explicitly observe q_goal / q_error / route_tangent.
Adding route-specific observation keys was the turning point.
```

This preserves the core action contract: the policy still outputs normalized joint-delta RL actions. The q_goal route is a target observation, not a command.

## 14. Results So Far

Baseline before route curriculum:

```text
full 483-waypoint route:
success rate: 0.0435
longest success prefix: 21
```

Route-observation sequential curriculum results:

```text
prefix_20:
success rate: 1.0
longest success prefix: 20

prefix_40:
success rate: 1.0
longest success prefix: 40

prefix_80:
success rate: 1.0
longest success prefix: 80
cumulative successful route distance: 1.120 m

prefix_120:
success rate: 1.0
longest success prefix: 120
cumulative successful route distance: 1.720 m
mean final position error: 0.00934 m
mean final orientation error: 0.02444 rad
```

Full-route probe using the prefix_120 model:

```text
target count: 483
success rate: 0.4741
longest success prefix: 170
cumulative successful route distance: 2.455 m
first failure index: 171
first failure reason: position
```

This satisfies the initial minimum, medium, and high targets:

- The longest prefix increased from 21 to well above 40.
- Chunk 1, indices 41-80, is no longer zero-success.
- Prefix 80 and prefix 120 sequential evals are stable.
- Full-route success is clearly above the original 0.0435 baseline.

The full 483-waypoint route is still not solved.

## 15. Known Failed Direction: Prefix 180 Forgetting

A prefix_180 fine-tune was attempted from the prefix_120 checkpoint. It looked acceptable under sampled training metrics, but failed the actual sequential-prefix test:

```text
prefix_180 fine-tune:
success rate over indices 1-180: 0.5667
longest success prefix: 1
first failure index: 2
first failure reason: position
```

This is a curriculum failure, not a proof that route curriculum stopped working. The run over-focused on later segment resets and forgot early prefix-start behavior.

The current best model is therefore:

```text
artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip
```

The prefix_180 model should not be used as the main checkpoint.

## 16. Anti-Forgetting Retry Result

An anti-forgetting prefix_180 config was added and run:

```text
config:
hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/route_curriculum_prefix180_routeobs_sequence2_antiforget.yaml

run:
artifacts/kinematic_phase1/route_curriculum/route_prefix180_routeobs_sequence2_antiforget_1m_001
```

It continued from the verified prefix_120 checkpoint and used more early-prefix replay:

```text
prefix_start_reset_ratio: 0.25
random_prefix_reset_ratio: 0.45
segment_reset_ratio: 0.20
recovery_reset_ratio: 0.10
learning_rate: 1.5e-5
```

Result:

```text
prefix_180 anti-forgetting retry:
sampled recent success rate: 0.9063
sampled recent route-ready hit rate: 0.9063
sequential actual-final-q success rate: 0.6000
longest success prefix: 1
first failure index: 2
first failure reason: position
```

This is an important negative result. Sampled training metrics were high, but actual sequential-prefix behavior still regressed immediately. The model often gets close transiently, but fails to preserve the early sequential route-ready position gate after the first waypoint.

The prefix_180 anti-forgetting model should not be used as the main checkpoint.

## 17. Next Curriculum Adjustment

The next prefix_180 attempt should preserve early-prefix behavior while adding later route coverage, but simple reset-ratio tuning is not enough. The next main design is teacher-anchored segment mastery.

Recommended changes:

- Lock the prefix_120 checkpoint as official teacher.
- Keep the prefix_120 checkpoint as the official best unless sequential eval improves.
- Add checkpoint selection based on sequential actual-final-q eval, not sampled route-window success alone.
- Consider an explicit early-prefix retention eval during training before accepting any prefix_180 candidate.
- Avoid treating segment-sampled success as sufficient evidence for promotion.
- Use prefix120 teacher action anchoring on route indices 1-120 while learning later route segments.

The next experiment should be treated as anti-forgetting prefix expansion:

```text
prefix_120 stable base
-> prefix_180 with early-prefix replay
-> accept only if sequential prefix from index 1 is preserved
```

The corresponding config has been added:

```text
hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/route_curriculum_prefix180_routeobs_sequence2_antiforget.yaml
```

Current best remains:

```text
artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip
```

## 18. Teacher-Anchored Segment121-180 Design

The next experiment should not immediately ask the model to run 1-180 sequentially. It should first test whether the model can learn the 121-180 local route distribution without destroying the verified 1-120 behavior.

Official teacher:

```text
artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip
```

Teacher dataset:

```text
artifacts/kinematic_phase1/route_curriculum/prefix120_teacher_anchor/teacher_route_anchor_dataset.npz
```

New tools:

- `hrl_trainer.kinematic_phase1.route.collect_route_teacher_rollout`
- `hrl_trainer.kinematic_phase1.route.teacher_anchor.RouteTeacherAnchorCallback`
- `hrl_trainer.kinematic_phase1.eval.eval_route_gate`
- optional `route.sequential_gate` support in `train_route_curriculum`

Training config:

```text
hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/route_segment121_180_teacheranchored.yaml
```

Reset mixture:

```text
50% segment 121-180
30% protected replay 80-120
20% route-start replay
```

Teacher anchor:

```text
candidate action ~= prefix120 teacher action on route_index <= 120
```

This is still RL fine-tuning. The teacher is not an IK controller and does not replace the policy. It only regularizes the early route so PPO updates do not erase the behavior that already works.

## 19. Sequential Gate Acceptance Rule

Every candidate that wants to replace the current best must pass a sequential gate:

```text
prefix20 sequential eval
prefix40 sequential eval
prefix80 sequential eval
prefix120 sequential eval
prefix180 sequential eval
optional full483 probe
```

Acceptance criteria:

```text
prefix120 longest_success_prefix >= 120
prefix120 success_rate ~= 1.0
prefix180 longest_success_prefix > 120
first_failure_index must not move before or equal to 120
full483 longest_success_prefix must not regress far below current best 170
```

If sampled training success is high but sequential prefix fails at index 2, the checkpoint is rejected.

## 20. Teacher-Anchored Experiment Result

The prefix120 teacher dataset was collected successfully:

```text
dataset:
artifacts/kinematic_phase1/route_curriculum/prefix120_teacher_anchor/teacher_route_anchor_dataset.npz

successful teacher indices: 1-120
sample count: 537
```

A short teacher-anchored smoke run was highly informative:

```text
run:
artifacts/kinematic_phase1/route_curriculum/route_segment121_180_teacheranchored_smoke_001

prefix20 success: 1.0
prefix40 success: 1.0
prefix80 success: 1.0
prefix120 success: 1.0
prefix180 success: 0.9444
prefix180 longest_success_prefix: 170
full483 success: 0.4741
full483 longest_success_prefix: 170
gate accepted: true
```

This validates the main idea: teacher anchoring can prevent the immediate index-2 collapse seen in the earlier prefix180 attempts.

However, a longer 1M teacher-anchored run again regressed:

```text
run:
artifacts/kinematic_phase1/route_curriculum/route_segment121_180_teacheranchored_1m_001

prefix20 success: 0.05
prefix120 success: 0.625
prefix180 success: 0.6444
longest_success_prefix: 1
first_failure_index: 2
gate accepted: false
rejection reasons:
- prefix120_retention_failed
- prefix180_did_not_expand_beyond_120
- prefix180_failed_before_or_at_120
```

Interpretation:

```text
teacher-anchor is promising, but training must use sequential-gated checkpoint selection / early stopping.
latest checkpoint is not safe.
```

The short accepted candidate is a useful experimental artifact, but the official best remains the prefix120 checkpoint until a longer run produces a gate-accepted model with a stronger full-route prefix.
