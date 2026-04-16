## Purpose
This document is a short executive summary of the V5.1 stage. It is intended for fast review by an advisor, teammate, or future self without requiring full access to the detailed experiment history.

# V5.1 Executive Summary

## Stage Objective
The V5.1 stage studied whether a stochastic SAC-based manipulation policy could be trained in Gazebo to approach a near-home target and, more importantly, whether that behavior could be consolidated into a usable deterministic controller.

## Initial Situation
The stage began with several overlapping problems:

- a target generation bug allowed sampled target `Z` values above the valid home height,
- late-stage runs often produced large and unstable motions,
- safety correction could dominate execution,
- replay and reward attribution were misaligned with executed action,
- and deterministic evaluation often appeared nearly motionless even when stochastic training occasionally produced useful behavior.

At that point, the system had too many possible failure sources to support a reliable interpretation.

## What Was Fixed
Several threads materially improved the system:

### 1. Safety and executed-vs-raw mismatch
- target generation was corrected so target `Z` does not exceed home,
- raw action and executed action were separated in logging and diagnostics,
- executed-vs-raw penalties and correction statistics were added.

**Result:** the earlier "bursting" and heavily corrected motion pattern became much less dominant. Safety is no longer the main explanation for current failure.

### 2. Reward shaping and near-goal stabilization
- shell bonuses, basin-like attraction, drift penalties, exit penalties, and related shaping terms were explored,
- eventually consolidated into `phase_a_bootstrap_v2`.

**Result:** stochastic training behavior improved meaningfully. The policy learned reliable outer-level approach and occasional deeper behavior. Reward shaping was therefore partially effective.

### 3. Removing confounders
- action stage was locked to `S0_B`,
- target stage was locked to `TC0`,
- step budget was reduced to `10`,
- `action_scale = 0.08` became the main base setting,
- multi-seed baseline validation was performed.

**Result:** curriculum was effectively ruled out as the main bottleneck.

### 4. Better evaluation
- deterministic post-training evaluation,
- periodic deterministic evaluation,
- fixed evaluation suites,
- best-checkpoint logic,
- early stopping,
- restore-best behavior,
- and gap evaluation
were added.

**Result:** the project could finally distinguish stochastic capability from deterministic capability. This turned out to be crucial.

## What Partially Worked but Did Not Solve the Core Problem

### SAC-internal stochastic-to-deterministic gap handling
Entropy annealing and SAC-internal solidification/distillation did make the deterministic mean action larger.

**However:** they did not convert that larger mean into stable deeper geometry. In practice, they increased mean action magnitude more than they improved deterministic control depth.

### Teacher-student deterministic extraction
A separate deterministic student was trained from teacher-generated executed-action datasets.

- Dataset v1 was too shallow:
  - `100` samples,
  - `9` inner,
  - `0` dwell.
- Dataset v2 was much larger:
  - `456` samples,
  - `29` inner,
  - `9` dwell,
  - but still dominated by outer samples.

Student v2 improved final retention relative to student v1, but still did not exceed the best teacher deterministic result.

## Current Best Quantitative Picture

### Best teacher deterministic result
- `true_outer_hit_rate = 0.2`
- `true_inner_hit_rate = 0.0`
- `true_dwell_hit_rate = 0.0`
- `true_basin_hit_rate = 0.2`
- `success_rate = 0.2`
- `mean_final_dpos ~= 0.0985`
- `regression_rate = 0.8`

### Best student deterministic result
- `true_outer_hit_rate = 0.2`
- `true_inner_hit_rate = 0.0`
- `true_dwell_hit_rate = 0.0`
- `true_basin_hit_rate = 0.2`
- `success_rate = 0.2`
- `mean_final_dpos ~= 0.1008`
- `regression_rate = 0.8`

The student improved over earlier student variants, but did not surpass the best teacher baseline.

## What Has Been Ruled Out
At this stage, the following are no longer the most accurate explanations for the observed limitation:

- safety correction as the dominant failure source,
- curriculum as the dominant failure source,
- the early target `Z` bug,
- broad instability or uncontrolled bursting as the dominant remaining issue,
- or simply "not enough action magnitude."

## Current Best Stage Conclusion
The most accurate interpretation is no longer "the system cannot approach the target."

The more accurate conclusion is:

- the stochastic teacher can learn useful outer-level approach behavior,
- occasional deeper behavior does exist,
- but that deeper behavior has not been consistently consolidated into a deterministic controller.

## Why the Stage Is Being Closed Here
V5.1 already produced a coherent stage result:

- broad instability was substantially reduced,
- stochastic approach behavior became real and repeatable,
- the remaining bottleneck was narrowed,
- and multiple attempts to consolidate deeper behavior into deterministic control did not break through the outer-level ceiling.

Further V5.1 iteration would likely add more variation without changing the central conclusion.

## Concluding Statement
We reduced the problem from broad system instability to a much narrower control bottleneck: the stochastic teacher can reliably learn outer-level approach behavior, but deeper inner-level behavior still cannot be consistently consolidated into a deterministic controller.
