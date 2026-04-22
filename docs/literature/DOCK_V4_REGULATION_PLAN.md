# Dock V4 Regulation Plan

## Purpose

This note records the design rationale for the next docking-stage rewrite in Phase 1B.
The goal is not to make docking a smaller version of the general reaching policy.
The goal is to turn docking into a local regulation problem:

- small bounded corrections
- deterministic convergence
- stable hold at the target pose

This document is intentionally narrow. It only covers the docking subproblem.

## Why the current docking line is being changed

Recent Phase 1B experiments established a consistent pattern:

- the controller can usually enter the strict pose region at least once
- position compression improved in some variants
- orientation is often already good enough
- but the controller does not reliably remain in the strict final pose
- long PPO training often re-inflates action magnitude instead of settling into low-gain regulation

The current issue is therefore not simple reaching. It is local regulation near the setpoint.

## Literature signals we are adopting

### 1. Hold / regulation should not be treated like ordinary reaching

Multiple robot RL papers after 2022 treat the near-goal phase differently from the approach phase.
Near the target, the objective shifts away from progress and toward stability, bounded control effort,
and staying in the desired state.

### 2. Low control effort near the target is part of the task

The literature repeatedly uses action or velocity penalties near the target, not just as generic smoothing,
but as a way to express that the correct terminal behavior is low-energy regulation.

### 3. Bounded corrective actions matter

For local tracking / correction problems, bounded residual corrections are a more natural structure than a
general free-moving policy. This is especially relevant for our docking Stage A task, where the policy should
make only small corrective moves around an already-correct pose.

### 4. Deterministic control is a better fit than entropy-seeking control

Our current docking Stage A objective is closer to setpoint regulation than exploration-driven skill acquisition.
This suggests a deterministic controller class is a better fit than PPO-style on-policy large-motion updates.
For the next version we therefore switch docking to TD3.

## Structural changes in Dock V4

Dock V4 changes more than reward weights. It changes the controller interpretation.

### 1. Dock becomes a bounded residual controller

The docking action is still a normalized joint delta command, but for docking mode it is additionally clipped by
an explicit residual-action bound before conversion to joint deltas.

This means:

- docking cannot issue large corrective swings
- docking is structurally biased toward small local regulation
- the task definition becomes closer to low-gain setpoint control

### 2. Dock Stage A uses TD3 instead of PPO

Dock Stage A is no longer treated as an on-policy exploration problem.
It is trained with TD3 as a deterministic actor-critic baseline better aligned with local regulation.

### 3. Reward becomes strict-zone-centric

Dock V4 keeps the strict-only philosophy:

- the main positive value lives inside the strict zone
- exact center is better than the strict boundary
- low action and low drift are rewarded only when the policy is already near the target
- leaving the strict zone remains explicitly bad

### 4. Reset distribution remains local

Stage A still starts at the exact goal state or in a very small neighborhood around it.
The task is not “how to reach the target from far away”.
The task is “how to remain at the target with small deterministic corrections”.

## Expected benefits

Dock V4 is intended to address the specific failure mode observed in the earlier runs:

- entering the strict region but failing to hold there
- drifting to the strict boundary
- developing unnecessarily large actions during long training

The intended effect is to force the controller to behave like a local stabilizer instead of a general docking policy.

## What success would look like

Dock V4 should improve these properties first:

- lower average action magnitude
- higher strict-pose final rate
- better final hold quality, not merely better one-time strict entry
- reduced regression after entering strict pose

Strict success rate is still important, but it is not the only measure.
The more important sign is whether the controller becomes a true local attractor around the target pose.

## Sources

- Guzman et al., *A Robotic Embodiment of Human-Like Motor Skills through Deep Reinforcement Learning*, IEEE RA-L, 2022:
  [PDF](https://www.luisjguzman.com/media/RAL_2022_Internet_of_Skills.pdf)
- Ishihara et al., *Learning a Reference Correction for Real Robot Manipulator Control with Deep Reinforcement Learning*, ICRA 2022:
  [arXiv](https://arxiv.org/abs/2203.07051)
- Li et al., *Pose Coordination Planning and Capture Strategy for Space Targets Based on Deep Reinforcement Learning*, Aerospace, 2024:
  [MDPI](https://www.mdpi.com/2226-4310/11/9/706)
- CALF / goal-reaching guarantees line, 2024:
  [arXiv](https://arxiv.org/abs/2409.14867)
