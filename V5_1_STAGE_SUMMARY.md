## Purpose
This document is the primary stage summary for V5.1. It records what was implemented, what was tested, what was ruled out, what partially worked, and why the V5.1 stage is being closed without further experimental extension.

# V5.1 Stage Summary

## Scope and Evidence Base
This summary is based on the current V5.1 codebase and on artifact outputs already generated during the stage. The main evidence sources are:

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/pipeline_e2e.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/sac_torch.py`
- `artifacts/v5_1/e2e/main_reward_bonly_baseline_s0b_tc0_act008_001/eval/deterministic_summary.json`
- `artifacts/v5_1/teacher_dataset/det_extract_v1/dataset_summary.json`
- `artifacts/v5_1/teacher_dataset/det_extract_v2/dataset_summary.json`
- `artifacts/v5_1/student/det_student_v1_eval/student_eval_summary.json`
- `artifacts/v5_1/student/det_student_v2_eval/student_eval_summary.json`

The document separates three levels of statement whenever possible:

- **Confirmed fact**: directly supported by code structure or artifact metrics.
- **Interpretation**: a technically grounded reading of those facts.
- **Current best conclusion**: the most accurate stage-level conclusion supported by the accumulated evidence.

## 1. Project Goal
The V5.1 stage aimed to train a manipulation policy that can drive the robot arm from near-home configurations toward a near-home target in Gazebo, using a stochastic SAC actor during training and evaluating whether the learned behavior can be consolidated into a useful deterministic controller.

The practical objective was not merely to obtain occasional stochastic proximity to the target, but to produce a controller that can:

- approach the target reliably,
- remain near the target instead of drifting away,
- and ultimately express that behavior in deterministic execution rather than only through exploration noise.

## 2. Current System Architecture

### 2.1 Training Loop
**Confirmed fact**

The training pipeline follows the structure:

`obs -> stochastic SAC actor -> executor clamp/projection -> Gazebo execution -> reward -> replay`

In V5.1, the actor-side execution can be summarized as:

`a_exec_before_safety = tanh(mu(obs) + exploration_std_scale * std(obs) * noise) * action_scale`

During deterministic evaluation, `noise = 0`.

### 2.2 Executed Action vs Proposed Action
**Confirmed fact**

The policy proposes an action, but the environment may execute a safety-modified action after clamp and/or projection. The replay path and diagnostics were extended so that raw action and executed action are both observable and can be compared explicitly.

### 2.3 Observation Structure
**Confirmed fact**

The main observation components are:

- joint positions `q`
- joint deltas `dq`
- end-effector pose error
- previous action

### 2.4 Post-Training Evaluation Infrastructure
**Confirmed fact**

V5.1 later added:

- deterministic post-training evaluation,
- periodic deterministic evaluation during training,
- fixed evaluation suites,
- gap diagnosis across deterministic and stochastic/noisy settings,
- best-checkpoint selection and restore-before-final-eval behavior.

### 2.5 Teacher-Student Extension
**Confirmed fact**

In the later 5.2 extension within the same stage, a second path was added:

`teacher artifacts -> teacher dataset -> deterministic student -> deterministic evaluation`

This path does not modify the SAC teacher training loop. It uses teacher-generated traces to build a supervised dataset and trains a separate deterministic policy by weighted regression to executed actions.

## 3. Main Failure Modes Encountered Early

### 3.1 Target Generation Bug
**Confirmed fact**

Early in the stage, target generation contained a bug in which the sampled target `Z` coordinate could exceed the home end-effector height, even though home represented the arm's upper vertical limit in the intended setup.

**Interpretation**

This injected invalid or misleading goals into training and evaluation, which confounded early runs.

### 3.2 Large Late-Training Motions and Safety Corrections
**Confirmed fact**

Early and mid-stage runs showed repeated patterns of:

- very large late-episode motions,
- erratic or "bursting" arm motion,
- frequent clamp/projection corrections,
- and poor alignment between proposed action and actual executed transition.

### 3.3 Deterministic Inactivity
**Confirmed fact**

The deterministic policy often appeared nearly motionless, while stochastic execution could still move the arm into useful regions.

**Interpretation**

This immediately suggested that some useful behavior existed in the learned stochastic policy, but that behavior had not been consolidated into the mean policy.

### 3.4 Reward/Transition Misalignment
**Confirmed fact**

Replay originally centered too strongly on raw action even though the environment transition was caused by executed action. This created an executed-vs-raw mismatch.

**Interpretation**

This mismatch can distort the action-effect relationship seen by the critic and actor, especially when safety correction is nontrivial.

## 4. Experimental Thread 1: Safety and Executed-vs-Raw Mismatch

### 4.1 Main Changes
**Confirmed fact**

This thread introduced and strengthened:

- target generation correction so sampled target `Z` does not exceed home,
- logging of `a_raw` and `a_exec`,
- logging of `delta_a`, `delta_norm`, `raw_norm`, `exec_norm`,
- clamp/projection/reject indicators,
- reward terms and diagnostics related to executed-vs-raw discrepancy,
- replay structures that preserve executed action information.

### 4.2 What It Addressed
**Interpretation**

The purpose was to stop training from learning on an incorrect assumption that raw action alone determined the resulting transition, and to reduce the pathological regime in which the actor proposes unrealistic motions that are heavily altered before execution.

### 4.3 Objective Outcome
**Confirmed fact**

This thread substantially reduced the earlier pattern of:

- late-stage "explosive" motion,
- frequent large corrections,
- and safety-dominated behavior.

### 4.4 Stage Interpretation
**Current best conclusion**

Safety correction and executed-vs-raw mismatch were important early-stage problems, but they are no longer the dominant bottleneck in the final state of V5.1.

## 5. Experimental Thread 2: Reward Shaping and Near-Goal Stabilization

### 5.1 Main Changes
**Confirmed fact**

This thread explored multiple reward shaping refinements, including:

- near-goal shell rewards,
- widened shell regions,
- stronger shell bonuses,
- attraction-style shaping that increases reward as distance decreases,
- local drift penalties,
- zone exit penalties,
- dwell break penalties,
- and eventually the more consolidated `phase_a_bootstrap_v2` profile.

### 5.2 What It Addressed
**Interpretation**

The aim was not simply to help the arm touch the vicinity of the target once, but to create a reward landscape that teaches:

- how to enter the basin,
- how to remain in the basin,
- and how to avoid drifting back out after partial success.

### 5.3 Objective Outcome
**Confirmed fact**

After these changes, stochastic training behavior became substantially healthier:

- the policy could more consistently enter the basin,
- outer-level approach became repeatable,
- and occasional deeper approach into inner or dwell regions began to appear.

### 5.4 What It Did Not Solve
**Confirmed fact**

These reward changes did not produce stable deterministic inner-level control.

### 5.5 Stage Interpretation
**Current best conclusion**

Reward shaping in V5.1 was not useless. On the contrary, it was sufficient to support meaningful stochastic approach behavior. However, it was not sufficient to consolidate that behavior into a deterministic controller that reliably penetrates beyond the outer region.

## 6. Experimental Thread 3: Locking Conditions and Removing Confounders

### 6.1 Main Changes
**Confirmed fact**

To reduce ambiguity, later experiments locked major conditions:

- action stage locked to `S0_B`,
- target stage locked to `TC0`,
- steps per episode reduced to `10`,
- `action_scale` converged to `0.08` as the most credible base setting,
- `exploration_std_scale` commonly held at `0.60`,
- and baseline behavior was checked across multiple seeds.

### 6.2 Objective Outcome
**Confirmed fact**

These locked-condition runs established that:

- curriculum changes were not the main driver of the later failure pattern,
- the chosen base configuration was not merely a one-off artifact,
- and the remaining issue persisted even after major confounders were reduced.

### 6.3 Stage Interpretation
**Current best conclusion**

By the time these runs were completed, curriculum had been largely demoted from "suspected root cause" to "not the primary bottleneck." The problem persisted under fixed `S0_B + TC0` conditions.

## 7. Experimental Thread 4: Deterministic Evaluation, Fixed Eval Suite, and Best Checkpoint Logic

### 7.1 Main Changes
**Confirmed fact**

This thread added:

- deterministic post-training evaluation,
- periodic deterministic evaluation during training,
- fixed evaluation suites,
- gap evaluation across noise levels,
- best-checkpoint logic,
- early stopping,
- and restore-best-before-final-eval behavior.

### 7.2 What It Clarified
**Interpretation**

This thread was essential because it separated two previously entangled questions:

- what the stochastic training policy can do,
- and what the final deterministic controller can do.

### 7.3 Objective Outcome
**Confirmed fact**

After this evaluation structure was in place, it became repeatedly observable that:

- stochastic train rollouts could show useful target approach behavior,
- while deterministic evaluation still lagged behind,
- and final deterministic performance could not be inferred from training behavior alone.

### 7.4 Stage Interpretation
**Current best conclusion**

This thread did not directly improve control, but it materially improved the correctness of interpretation. It made the central gap visible: stochastic capability existed, but deterministic consolidation remained weak.

## 8. Experimental Thread 5: Handling the Stochastic-to-Deterministic Gap Inside SAC

### 8.1 Main Changes
**Confirmed fact**

Several interventions were tested inside the SAC framework:

- entropy annealing with `A -> B -> C`,
- a reduced `B-only` annealing schedule,
- SAC-internal solidification and distillation variants,
- including `B-only baseline`,
- `B-only + solidify v1`,
- and `B-only + solidify v2`.

### 8.2 Objective Outcome
**Confirmed fact**

Across these variants, a consistent pattern emerged:

- deterministic mean action magnitude increased,
- det/stoch action ratios increased,
- but deterministic geometry did not improve proportionally.

The core outcome was not a deeper controller, but a larger mean action.

### 8.3 What Was Not Achieved
**Confirmed fact**

These runs did not establish stable deterministic inner-level performance.

### 8.4 Stage Interpretation
**Current best conclusion**

Within SAC, these methods were partially effective at making the mean policy less inert, but they did not solve the core problem. They scaled the mean more than they improved control depth or geometric precision.

## 9. Experimental Thread 6: Teacher-Student Deterministic Extraction

### 9.1 Main Design
**Confirmed fact**

The later 5.2 extension introduced:

- a teacher dataset builder from existing SAC artifacts,
- a deterministic student trainer,
- and a student evaluation pipeline.

The student was trained by weighted supervised regression to executed action, not by online actor-critic updates.

### 9.2 Motivation
**Interpretation**

The goal was to stop asking one SAC actor to serve simultaneously as:

- an exploratory stochastic policy,
- and the final deterministic controller.

Instead, SAC was explicitly treated as a teacher that generates exploratory behavior, while a second model tried to extract a deterministic controller from the teacher's best executed actions.

### 9.3 Teacher-Student v1
**Confirmed fact**

Teacher dataset v1 contained:

- total samples: `100`
- elite: `59`
- strong: `41`
- outer: `68`
- inner: `9`
- dwell: `0`

Student v1 achieved:

- `true_outer_hit_rate = 0.2`
- `true_inner_hit_rate = 0.0`
- `true_dwell_hit_rate = 0.0`
- `success_rate = 0.2`
- `mean_final_dpos ~= 0.112`
- `regression_rate = 0.8`

### 9.4 Teacher-Student v2
**Confirmed fact**

A larger teacher collection campaign was then run, adding:

- `main_reward_collect_bonly_s0b_tc0_act008_seed101`
- `main_reward_collect_bonly_s0b_tc0_act008_seed202`
- `main_reward_collect_solidify_s0b_tc0_act008_seed303`
- `main_reward_collect_solidify_s0b_tc0_act008_seed404`

These were merged with the earlier teacher runs to form dataset v2:

- total samples: `456`
- elite: `316`
- strong: `140`
- dwell: `9`
- inner: `29`
- outer: `268`
- outside: `150`

Student v2 achieved:

- `true_outer_hit_rate = 0.2`
- `true_inner_hit_rate = 0.0`
- `true_dwell_hit_rate = 0.0`
- `true_basin_hit_rate = 0.2`
- `success_rate = 0.2`
- `mean_final_dpos ~= 0.1008`
- `regression_rate = 0.8`

### 9.5 Teacher-Student Interpretation
**Interpretation**

Teacher-student extraction was not completely ineffective. The larger v2 dataset improved retention relative to student v1. However, it still did not produce a deterministic controller that exceeded the best teacher deterministic result.

### 9.6 Stage Conclusion for This Thread
**Current best conclusion**

The teacher-student path confirmed that simply increasing the amount of teacher data and training a separate deterministic regressor was not sufficient to reliably create inner-level deterministic control. The deeper behavior existed in traces, but not in a sufficiently concentrated or consolidatable form.

## 10. What Has Been Ruled Out

### 10.1 Ruled Out as Primary Bottlenecks
**Current best conclusion**

Based on the accumulated experiments, the following are no longer the leading explanations for the current failure mode:

- safety correction as the dominant problem,
- curriculum as the dominant problem,
- the target `Z` generation bug as the dominant problem,
- large clamp/projection-heavy instability as the dominant problem,
- merely increasing mean action magnitude as a sufficient solution,
- merely increasing dataset size as a sufficient solution.

### 10.2 What This Means
**Interpretation**

The stage has already removed many broad system-level confounders. The remaining problem is narrower and more specific than it was at the start.

## 11. Current Best Teacher Result

### 11.1 Metric Summary
**Confirmed fact**

The current best teacher deterministic result, represented by the best baseline deterministic evaluation, is approximately:

- `true_outer_hit_rate = 0.2`
- `true_inner_hit_rate = 0.0`
- `true_dwell_hit_rate = 0.0`
- `true_basin_hit_rate = 0.2`
- `success_rate = 0.2`
- `mean_final_dpos ~= 0.0985`
- `regression_rate = 0.8`

### 11.2 Interpretation
**Interpretation**

The teacher deterministic policy is not inert. It can reach the outer region and occasionally succeed. However, it remains constrained to outer-level behavior and does not show stable inner-level deterministic control.

## 12. Current Best Student Result

### 12.1 Student v1
**Confirmed fact**

Student v1 matched the teacher's outer-level deterministic performance but was worse in final retention.

### 12.2 Student v2
**Confirmed fact**

Student v2 improved over v1 in retention:

- `mean_final_dpos` improved from about `0.112` to about `0.1008`
- `mean_final_minus_min` also improved

However, student v2 still did not exceed the teacher's best deterministic result:

- outer remained `0.2`
- inner remained `0.0`
- success remained `0.2`
- regression remained `0.8`
- final retention still remained slightly worse than the best teacher deterministic reference

### 12.3 Stage Interpretation
**Current best conclusion**

The best student currently behaves as a somewhat cleaner outer-level controller, not as a controller that has broken through the deterministic inner-level bottleneck.

## 13. Core Unresolved Bottleneck

### 13.1 Narrowed Problem Definition
**Confirmed fact**

At the current stage:

- the stochastic teacher can reliably approach the target basin,
- occasional deeper behavior does appear,
- but deterministic policies remain stuck at the outer level.

### 13.2 Interpretation
**Interpretation**

The project is no longer limited by broad instability. It is now limited by a consolidation problem:

- deeper exploratory behavior exists,
- but it is too sparse, too weakly concentrated, or too poorly consolidated to become stable deterministic inner-level control.

### 13.3 Current Best Conclusion
**Current best conclusion**

The central unsolved bottleneck is not whether the system can approach the target at all. The bottleneck is whether deeper behavior can be consistently consolidated into a deterministic controller. At the end of V5.1, that has not yet been achieved.

## 14. Why V5.1 Is Being Closed at This Stage

### 14.1 Reason for Closure
**Current best conclusion**

V5.1 is being closed here because the stage has already produced a clear and coherent experimental answer:

- broad system instability was substantially reduced,
- safety mismatch was addressed enough to stop dominating the outcome,
- reward shaping produced useful stochastic approach behavior,
- curriculum confounds were reduced,
- evaluation instrumentation became reliable,
- and the remaining failure mode has been narrowed to a specific deterministic consolidation bottleneck.

Continuing to add more V5.1 experiments at this point would likely create more run-level variation without changing the core stage conclusion.

### 14.2 What Has Been Achieved by Closing Here
**Interpretation**

Closing the stage now preserves a clean technical result. It avoids turning V5.1 into an open-ended collection of partially overlapping experiments and preserves a coherent narrative about what was learned.

## 15. Recommended Interpretation of V5.1 as a Stage Result

V5.1 should be interpreted as a successful narrowing stage rather than as a final control solution.

### 15.1 What V5.1 Successfully Established
**Confirmed fact**

V5.1 established that:

- the pipeline can produce meaningful stochastic target approach behavior,
- safety-related instability can be substantially reduced,
- the system can be instrumented and evaluated in a much more reliable way,
- and the dominant unresolved issue is specifically deterministic consolidation.

### 15.2 What V5.1 Did Not Establish
**Confirmed fact**

V5.1 did not establish:

- robust deterministic inner-level control,
- stable deterministic dwell behavior,
- or a deterministic controller that clearly surpasses the best teacher baseline.

### 15.3 Final Concluding Statement
We reduced the problem from broad system instability to a much narrower control bottleneck: the stochastic teacher can reliably learn outer-level approach behavior, but deeper inner-level behavior still cannot be consistently consolidated into a deterministic controller.
