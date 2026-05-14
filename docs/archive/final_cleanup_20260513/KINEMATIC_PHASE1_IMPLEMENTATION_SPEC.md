## Purpose
This document freezes the Phase 1 implementation scope for the kinematic-first research direction. It is the concrete companion specification to `KINEMATIC_FIRST_RESEARCH_PLAN.md`, and it defines what Phase 1 should include, what it should explicitly exclude, and what counts as completion.

# Phase 1 Implementation Spec
## Gymnasium Kinematic Arm Environment + Stable-Baselines3 Baselines

## 1. Objective

Phase 1 establishes a clean research stack for a 6–7 DoF arm using:

- Gymnasium as the custom environment interface,
- Stable-Baselines3 as the RL baseline library,
- and pure forward-kinematics state transition, without Gazebo or real hardware.

This phase is only for:

1. reach,
2. near-goal stabilization.

This phase is not for:

- waypoint carrying,
- tray dynamics,
- real robot deployment,
- contact-rich physics,
- or ROS / Gazebo integration.

The purpose is to create a high-throughput kinematic training setup that is easy to iterate on.

## 2. Why This Phase Exists

The current Gazebo-centric RL pipeline is too heavy for fast research iteration.

At this stage, kinematics are already known, so the main goal is not realistic physics but fast learning of basic arm skills.

The Phase 1 research loop should therefore emphasize:

- clean action/observation definition,
- forward-kinematics-based transition,
- fast repeated training runs,
- and clean deterministic evaluation.

## 3. Framework Decisions

### 3.1 Environment

Use a custom Gymnasium environment API.

### 3.2 RL Library

Use Stable-Baselines3.

### 3.3 Policy Type

Use `MultiInputPolicy`, because the observation will use a `Dict` space.

### 3.4 Initial Baseline Algorithms

Provide training entry points for:

1. TD3
2. PPO

SAC can be added later for comparison, but Phase 1 should begin with TD3 and PPO.

## 4. Scope of Phase 1

### Task A: Reach

Move the end-effector toward a target pose.

### Task B: Stabilize

Once near the target, remain near the target for multiple consecutive steps.

Phase 1 does **not** include:

- waypoint following,
- tray carrying,
- supervised learning,
- phase-2 task logic.

## 5. Observation / Action Contract

The contract should be defined once and kept stable across later phases.

The design rule is:

**do not invent a new contract later.**

## 5.1 Action Space

Use:

- normalized continuous joint-delta command,
- shape `(n_joints,)`,
- with current default `n_joints = 7`.

Interpretation:

- `delta_q_cmd = action * joint_delta_limit_per_step`

Recommended Gymnasium space:

- `Box(low=-1, high=1, shape=(7,), dtype=np.float32)`

## 5.2 Observation Space

Use Gymnasium `Dict` space with the following keys:

- `q`: normalized joint positions, shape `(7,)`
- `dq`: normalized joint deltas / velocity-like term, shape `(7,)`
- `prev_action`: previous normalized action, shape `(7,)`

- `goal_pos_err`: current EE -> final goal position error, shape `(3,)`
- `goal_ori_err`: current EE -> final goal orientation error, shape `(3,)`

- `wp_pos_err`: current EE -> current waypoint position error, shape `(3,)`
- `wp_ori_err`: current EE -> current waypoint orientation error, shape `(3,)`
- `next_wp_pos_err`: current EE -> next waypoint position error, shape `(3,)`
- `next_wp_ori_err`: current EE -> next waypoint orientation error, shape `(3,)`

- `task_type`: one-hot vector, shape `(3,)`
- `mode_flag`: one-hot vector, shape `(4,)`
- `progress`: task progress features, shape `(3,)`
- `joint_limit_margin`: normalized joint-limit margin, shape `(7,)`

## 5.3 Phase 1 Field Values

In Phase 1:

- `task_type = [1, 0, 0]`
- waypoint-related fields (`wp_*`, `next_wp_*`) are zero-filled, or aliased to goal if that is cleaner
- `mode_flag` can indicate `approach` or `stabilize`, but the shape must already exist
- `progress` can contain:
  - normalized episode progress,
  - normalized dwell counter,
  - one reserved scalar initialized to `0`

The important rule is:

**keep the observation schema stable even if some fields are not used yet.**

## 6. Environment Behavior

Implement a pure kinematic transition environment.

### 6.1 State Update

At each step:

1. read action,
2. map action to `delta_q_cmd`,
3. clip to per-step delta limits and joint limits,
4. update joint configuration,
5. run forward kinematics,
6. compute the end-effector pose,
7. compute next observation,
8. compute reward,
9. determine termination / truncation.

### 6.2 No Dynamics in Phase 1

Do not simulate:

- mass,
- inertia,
- collisions,
- friction,
- torque,
- latency.

Phase 1 is a kinematic environment only.

## 7. Reward Design for Phase 1

Reward should remain simple and interpretable.

### 7.1 Main Terms

Use a reward with these components:

- position error reduction,
- orientation error reduction,
- near-goal bonus,
- dwell bonus,
- drift penalty,
- action smoothness penalty,
- joint-limit penalty.

### 7.2 Design Goal

The reward should support:

- moving closer,
- staying near the goal,
- avoiding excessive oscillation,
- avoiding persistent pressure against joint limits.

Phase 1 should **not** recreate the old complex V5.1 reward tree.

## 8. Termination Logic

### 8.1 Success

Success should occur if:

- position error is below threshold,
- orientation error is below threshold,
- and optionally the condition is sustained for `N` consecutive steps.

### 8.2 Truncation

Truncate if:

- the maximum episode step limit is reached.

### 8.3 Optional Early Failure

Optional early failure is allowed for:

- numerical invalid state,
- impossible post-clip joint-limit violation.

However, Phase 1 failure logic should remain minimal.

## 9. Deterministic Evaluation

Phase 1 must include deterministic evaluation from day one.

Implement:

- a fixed evaluation suite,
- a deterministic rollout runner,
- an evaluation summary writer.

### 9.1 Required Metrics

At minimum, record:

- success rate,
- mean final position error,
- mean final orientation error,
- near-goal hit rate,
- dwell success rate,
- regression / drift rate,
- average action magnitude.

### 9.2 Separation Rule

Training and evaluation must remain separate.

The purpose is to compare:

- training reward trends,
- deterministic policy quality.

## 10. Suggested Repository Structure

Create a dedicated Phase 1 module tree, for example:

- `src/kinematic_phase1/`
  - `envs/`
    - `arm_kinematic_env.py`
    - `observation_builder.py`
    - `reward_fn.py`
    - `termination.py`
    - `spaces.py`
  - `kinematics/`
    - `fk_interface.py`
    - `joint_limits.py`
    - `pose_utils.py`
  - `training/`
    - `train_td3.py`
    - `train_ppo.py`
    - `policy_config.py`
    - `callbacks.py`
  - `eval/`
    - `eval_deterministic.py`
    - `fixed_eval_suite.py`
    - `metrics.py`
  - `configs/`
    - `phase1_default.yaml`
    - `td3_default.yaml`
    - `ppo_default.yaml`

Another naming scheme is acceptable, but the design should remain similarly modular.

## 11. Implementation Tasks

### Task 1 — Kinematic Environment Shell

Implement a Gymnasium environment class that:

- inherits from `gymnasium.Env`,
- defines action space and observation space,
- supports `reset()` and `step()`,
- uses FK-driven transition.

### Task 2 — Observation Builder

Centralize observation construction in one module.

Do not hand-build observation dictionaries in multiple places.

### Task 3 — Reward Function

Centralize reward computation in one module.

Return:

- scalar reward,
- reward component breakdown in `info`.

### Task 4 — Termination Logic

Implement success / truncation / optional failure checks in one module.

### Task 5 — Fixed Evaluation Suite

Add a reusable deterministic evaluation set with:

- fixed seeds,
- fixed targets,
- reproducible evaluation episodes.

### Task 6 — SB3 Training Entry Points

Implement:

- `train_td3.py`
- `train_ppo.py`

Both should:

- create the environment,
- train the model,
- save checkpoints,
- save logs,
- run deterministic post-train evaluation.

### Task 7 — Config Support

Use simple YAML or argparse-based configuration.

Support at minimum:

- joint count,
- step-size limits,
- episode length,
- reward thresholds,
- success thresholds,
- algorithm hyperparameters.

### Task 8 — Metrics and Reports

Write compact JSON summaries after training, including:

- training config,
- final metrics,
- best-checkpoint metrics if applicable,
- deterministic evaluation summary.

## 12. Initial Defaults

Use practical defaults such as:

- joint count = 7,
- episode length = 50–100 steps,
- action = normalized joint-delta command,
- deterministic evaluation always enabled after training.

For PPO and TD3:

- start with SB3-compatible default templates,
- do not over-tune on day one.

The first goal is a working baseline, not an optimized benchmark.

## 13. Acceptance Criteria

Phase 1 is complete when all of the following are true:

1. a custom Gymnasium kinematic arm environment exists and runs correctly,
2. TD3 training runs end-to-end,
3. PPO training runs end-to-end,
4. deterministic post-train evaluation is implemented,
5. results are saved in a reproducible format,
6. the observation schema already includes `task_type` and future-compatible fields even if unused in Phase 1,
7. the codebase is modular enough that Phase 2 can reuse the same observation/action contract.

## 14. Non-Goals

The following are explicitly out of scope for Phase 1:

- waypoint dataset builder,
- supervised training,
- tray carrying,
- ROS integration,
- Gazebo integration,
- real robot interface,
- contact dynamics,
- image observations,
- custom transformer architectures.

## 15. Final Note

The purpose of this phase is to produce a fast, clean, reproducible kinematic training stack.

The priority is:

- simple,
- modular,
- fast to iterate,
- deterministic to evaluate.

Do not pull old V5.1 complexity into this new stack unless a later validation stage clearly requires it.
