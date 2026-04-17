## Purpose
This document captures the next-stage research direction after formally closing the Gazebo-centered V5.1 / V5.2 stage. It is a restructuring memo, not an implementation patch. Its purpose is to explain why the main research loop is being moved away from heavy Gazebo/SAC iteration and toward a kinematic-first training stack.

# Kinematic-First Research Plan

## 1. Decision Summary

The current decision is to stop extending V5.1 / V5.2 as the primary experimental line and to move the main research loop into a pure kinematic simulation setting.

This decision does **not** mean the previous V5.1 / V5.2 work was unsuccessful or without value. The opposite is true: the earlier stage was valuable precisely because it narrowed the problem from broad system instability to a much more specific control bottleneck.

The key conclusion from the previous stage is:

- the problem is no longer dominated by system instability,
- it is no longer dominated by safety mismatches,
- it is no longer the case that reward shaping is entirely ineffective,
- but the current Gazebo-centered RL stack is not an efficient research environment for studying arm skill acquisition under known kinematics.

In practical terms, we already know:

- the forward kinematics,
- the joint limits,
- much of the task geometry,
- and the fact that not every behavior needs to be discovered from scratch through heavy online reinforcement learning.

What is needed now is:

- fast iteration,
- clean environment contracts,
- high training throughput,
- and a research loop that isolates learning questions from simulator/system integration issues.

## 2. One-Sentence Version

The new direction is to first build a high-throughput Gymnasium-based kinematic arm environment for reach and stabilization, then treat waypoint-following and orientation-constrained tasks as supervised learning problems, and only after that return to Gazebo or real execution for validation and deployment.

## 3. Why the Direction Is Being Changed

## 3.1 The Main Problem Has Changed

During V5.1 / V5.2, the project progressively ruled out several early-stage confounders:

- target-generation bugs,
- severe late-stage instability,
- excessive safety correction as the dominant failure mode,
- curriculum as the main bottleneck,
- and fully ineffective reward shaping.

That narrowing was important. It showed that the remaining issue is not a generic “the system does not work” problem. The remaining issue is that the current end-to-end Gazebo/SAC research loop is too expensive and too slow for the kind of skill-learning questions now being studied.

The current bottleneck is not primarily:

- unknown kinematics,
- unknown action constraints,
- or completely unconstrained behavior synthesis.

Instead, the main bottleneck is that the current research stack forces too many things into a single loop:

- simulator integration,
- safety/executor behavior,
- online RL instability,
- stochastic-to-deterministic consolidation,
- and task learning itself.

That coupling makes iteration slow and interpretation harder than it needs to be.

## 3.2 Not Every Problem Should Be Solved as RL

The tasks under consideration now can be separated into at least two categories.

### Category A: Skill Acquisition

Examples:

- move the arm from its current joint state toward a target pose,
- reach the target,
- remain near the target,
- stabilize instead of drifting away.

This category is naturally aligned with:

- skill acquisition,
- short-horizon control learning,
- and clean feedback-driven policy improvement.

### Category B: Structured Task Execution

Examples:

- follow known waypoints,
- maintain a required end-effector orientation,
- execute a tray-carrying or path-following behavior with a known structure.

This category is naturally aligned with:

- tracking,
- imitation,
- supervised learning,
- or structured optimization.

Treating both categories as one monolithic RL problem obscures the learning objective and raises the cost of every experiment. The new direction therefore separates them by design.

## 3.3 Known Kinematics Favors a Kinematic Research World

At the current stage, the main research questions are not:

- contact mechanics,
- friction modeling,
- sensor latency,
- or physically rich dynamic interaction.

The main questions are closer to:

- how the arm reaches a target,
- how it stabilizes near the target,
- how it follows waypoints,
- and how task behavior can be structured cleanly.

These questions can be studied effectively in a pure kinematic world first.

That is important because a pure kinematic environment allows:

- exact forward transitions under known geometry,
- fast reset and step throughput,
- repeatable controlled experiments,
- and much cleaner comparisons across algorithms and task decompositions.

## 3.4 Gazebo and Real Execution Should Become Validation Stages

When Gazebo is used as the main training environment too early, the research loop becomes dominated by:

- executor limits,
- runtime orchestration,
- simulator speed,
- integration bugs,
- reset costs,
- and other system-level noise.

The new plan explicitly changes that role:

- Gazebo is no longer the primary training battlefield,
- it becomes a validation and deployment stage.

This means the learning problem should first be solved in a cheaper and cleaner environment, and only later transferred back into the full execution stack.

## 4. High-Level Architecture of the New Plan

The new plan is intentionally decomposed into three phases.

## Phase 1: Gymnasium + Stable-Baselines3 for Basic Kinematic Skills

### 4.1 Objective

The first phase establishes a pure kinematic arm environment in Gymnasium and uses it to learn two basic skills:

- reach,
- stabilize.

The task definition at this stage is deliberately simple:

- input the current joint state,
- input the target pose,
- output a joint-delta action,
- propagate the next state through forward kinematics and simple transition rules,
- train with a clean, interpretable reward.

### 4.2 What Phase 1 Is For

Phase 1 is **not** meant to be a full final task benchmark. It is meant to establish:

- a stable environment contract,
- a fast training loop,
- a deterministic policy baseline,
- and a clear evaluation pipeline.

It is a research accelerator, not the final deployment condition.

### 4.3 Why Gymnasium

Gymnasium is chosen here because the project needs a clean environment API, not a heavy simulator.

Its role is to provide:

- well-defined `reset / step / observation / reward / done`,
- compatibility with batch experimentation,
- deterministic evaluation,
- and easy algorithm substitution.

This makes it a good wrapper for a custom kinematic arm environment.

### 4.4 Why Stable-Baselines3

Stable-Baselines3 is used to provide quick algorithmic baselines instead of rebuilding another RL training framework from scratch.

Its role is to:

- provide standard baselines such as TD3, PPO, or SAC,
- allow fast algorithm comparison in the kinematic setting,
- and avoid spending more research time on infrastructure rather than behavior learning.

The immediate value of SB3 is speed and comparability, not novelty.

### 4.5 Expected Outputs from Phase 1

By the end of Phase 1, the project should have:

- a pure kinematic Gymnasium arm environment,
- a fixed observation/action schema,
- a baseline reach policy,
- a baseline stabilization policy,
- a deterministic evaluation tool,
- and a fixed evaluation suite.

### 4.6 Success Criteria for Phase 1

Success at this stage does **not** mean solving the final tray task. It means:

- the arm can reliably reach target poses in the kinematic world,
- deterministic evaluation is stable and interpretable,
- near-goal stabilization is clearly better than random or naive control,
- and training throughput is much higher than the Gazebo-based loop.

## Phase 2: Supervised Learning for Waypoint and Orientation-Constrained Tasks

### 5.1 Objective

The second phase introduces the actual structured tasks, such as:

- tray carrying,
- waypoint following,
- fixed end-effector orientation maintenance,
- and other path-conditioned task behaviors.

This phase deliberately avoids treating the whole task as unconstrained exploratory RL.

### 5.2 Why Phase 2 Is Not Pure RL by Default

Phase 2 is different from Phase 1.

Phase 1 asks:

- how to approach a target,
- how to stay near a target.

Phase 2 asks:

- how to execute a structured behavior under known constraints,
- how to follow a path or waypoint sequence,
- how to respect orientation requirements.

If the task already has:

- known waypoints,
- known task flags,
- known orientation targets,
- and potentially reasonable expert trajectories,

then it resembles:

- supervised learning,
- imitation learning,
- or structured tracking,

more than it resembles unconstrained exploration.

### 5.3 Main Components of Phase 2

Phase 2 is expected to introduce:

- a waypoint dataset builder,
- a supervised trainer,
- and a task-conditioned policy.

The model input is expected to include:

- joint state,
- end-effector error,
- current waypoint error,
- next waypoint error,
- task type,
- and progress signals.

The output should remain the same as in Phase 1:

- normalized joint-delta action.

### 5.4 Critical Design Principle for Phase 2

The most important rule is:

**the observation/action contract should not change between phases.**

That means:

- Phase 1 should already reserve fields such as `task_type`, `mode_flag`, `waypoint_error`, and `progress`,
- even if some of them are zero-filled or inactive at first.

This makes later fine-tuning and reuse much cleaner, and avoids restarting the entire stack every time the task becomes more structured.

### 5.5 Expected Outputs from Phase 2

By the end of Phase 2, the project should have:

- a waypoint-conditioned dataset,
- a supervised waypoint-following policy,
- path-following evaluation,
- orientation-stability evaluation,
- and task-conditioned behavior separation.

### 5.6 Success Criteria for Phase 2

Success at this stage means:

- the arm can follow waypoints in the kinematic world,
- end-effector orientation remains stable when required,
- motion stays reasonably smooth,
- and different task types produce different appropriate behaviors.

## Phase 3: Return to Gazebo / Real Execution for Validation

### 6.1 Objective

The third phase brings the learned behavior back into:

- Gazebo,
- ROS/controller execution,
- and potentially later real robot runtime.

This phase is not intended to rediscover the behavior from scratch. It is intended to check:

- transfer gap,
- safety compatibility,
- controller-interface compatibility,
- and practical execution quality.

### 6.2 Role of Phase 3

Phase 3 is a validation and deployment stage.

This means:

- the main learning research should happen in Phases 1 and 2,
- the full execution stack should be used to verify transferability,
- and any remaining gap should be studied as a transfer/integration problem rather than as a reason to keep all training in Gazebo.

### 6.3 Metrics for Phase 3

The validation stage should focus on:

- final position and orientation error,
- waypoint tracking error,
- clamp/correction rate,
- latency impact,
- drift after reaching,
- and measurable differences between the kinematic world and Gazebo/runtime execution.

### 6.4 Success Criteria for Phase 3

Success at this stage means:

- policies learned in the kinematic world can function in the real execution stack,
- safety remains manageable,
- transfer degradation is understandable and analyzable,
- and the project does not need to restart training from scratch in Gazebo.

## 7. Cross-Phase Design Principle

The most important architectural rule across all three phases is:

**do not redefine the observation/action interface at every phase change.**

The unified contract should be designed from the start to support all phases.

### 7.1 Action Contract

Action should remain:

- normalized joint-delta command.

### 7.2 Observation Contract

Observation should reserve fields such as:

- task type,
- mode flag,
- goal error,
- waypoint error,
- progress,
- joint-limit margin.

This enables:

- simpler phase-to-phase continuity,
- more realistic fine-tuning paths,
- and less structural refactoring as task complexity increases.

## 8. How This Differs from the Old V5.1 / V5.2 Direction

The old direction was centered on a heavy RL loop that attempted to solve many problems at once:

- reward design,
- safety handling,
- exploration control,
- deterministic extraction,
- teacher-student consolidation,
- and full execution integration.

That direction was valuable because it revealed the actual bottleneck, but it also coupled research and systems work too tightly.

The new direction changes that decomposition:

- separate known geometry from unknown physics,
- separate exploratory skill learning from path-following skill learning,
- separate research training from deployment validation.

In plain terms:

- the old direction tried to build the whole structure at once,
- the new direction builds the foundation, the skeleton, and the validated transfer path separately.

## 9. What Is Intentionally De-Prioritized Now

This restructuring also means the following are no longer the primary research battlefield at this stage:

- Gazebo as the main training environment,
- SAC as the only core learning algorithm,
- solving reach, stabilize, waypoint following, and deployment behavior inside one monolithic RL loop,
- and continuing to micro-tune deterministic consolidation inside the old V5.1 / V5.2 architecture.

This is not a claim that those topics will never matter again. The point is that they are not the right center of gravity for the next research phase.

## 10. What the New Plan Is Actually Trying to Achieve

The purpose of the new plan is not to “just play in Gymnasium.”

The actual long-term goal is to establish a more efficient research path:

1. learn to reach in a fast kinematic world,
2. learn to stabilize in that same environment,
3. learn to follow structured waypoint/orientation tasks under supervised or imitation-style training,
4. and then transfer the resulting behavior back to Gazebo or real execution for validation.

In this view:

- the kinematic Gymnasium phase is a research accelerator,
- Gazebo and real execution remain important,
- but they move later in the pipeline where they are more useful and less disruptive.

## 11. Final Position Statement

The project is therefore moving away from a Gazebo-first and SAC-first research loop. The next stage will use a Gymnasium-based pure kinematic arm environment to learn reaching and stabilization efficiently, then handle waypoint-following and orientation-constrained tasks as supervised learning problems, and only after that return to Gazebo or real execution for validation.

This change is a deliberate restructuring of the research workflow, not a rejection of the value of the previous stage.
