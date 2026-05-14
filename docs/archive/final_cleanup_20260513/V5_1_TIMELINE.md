## Purpose
This document records the main V5.1 technical timeline. It is organized by problem-narrowing phases rather than by every individual run, so that the evolution of the stage can be understood quickly and accurately.

# V5.1 Timeline

## Phase 0. Initial Broad Failure State
The early V5.1 state contained multiple overlapping issues at the same time:

- target generation bug in the vertical dimension,
- large late-training motions,
- frequent safety correction,
- raw-action vs executed-action mismatch,
- weak near-goal retention,
- and deterministic policies that often appeared almost motionless.

At this point, the system had too many simultaneous uncertainties for a clean interpretation.

## Phase 1. Correct the Most Obvious System-Level Faults
The first phase focused on system hygiene and traceability:

- target `Z` generation was corrected so sampled targets do not exceed home height,
- raw and executed actions were separated,
- correction diagnostics and penalties were added,
- replay and logging were expanded around `a_raw`, `a_exec`, `delta_norm`, clamp, projection, and rejection.

### Outcome
- broad motion instability was reduced,
- large correction-dominated behavior became less central,
- safety mismatch was demoted from "primary cause" to "important but no longer dominant."

## Phase 2. Improve Reward Shaping for Approach and Retention
Once catastrophic instability was reduced, attention shifted to the reward landscape:

- shell rewards were added and widened,
- attraction-style shaping was strengthened,
- drift penalties and zone-exit penalties were introduced,
- multiple stabilization variants were tested,
- and the stage converged toward `phase_a_bootstrap_v2`.

### Outcome
- stochastic training began to show repeatable basin entry,
- outer-level approach became more stable,
- occasional deeper inner/dwell behavior appeared,
- but deterministic policies still did not reliably stabilize inside the deeper region.

## Phase 3. Lock Conditions and Remove Experimental Confounders
To avoid confusing reward effects with environment shifts:

- action stage was locked to `S0_B`,
- target stage was locked to `TC0`,
- step budget was reduced to `10`,
- `action_scale = 0.08` became the main working base,
- multiple seed baselines were run.

### Outcome
- curriculum was no longer a credible primary explanation,
- the remaining failure mode persisted even under fixed conditions,
- the base setting became substantially more trustworthy.

## Phase 4. Build a Reliable Evaluation and Selection Layer
The next major step was to improve observability:

- deterministic post-train evaluation was added,
- periodic deterministic evaluation was added,
- fixed evaluation suites were introduced,
- best-checkpoint logic was added,
- early stopping and resume-best behavior were added,
- gap evaluation across deterministic and noisy execution was added.

### Outcome
This phase did not itself improve policy quality, but it changed the quality of interpretation:

- train stochastic behavior and final deterministic behavior could finally be separated,
- and the project could now state clearly that stochastic capability existed before deterministic consolidation did.

## Phase 5. Try to Consolidate Behavior Inside SAC
The next question became:

Can SAC itself be adjusted so that the useful stochastic behavior moves into the deterministic mean policy?

The main attempts were:

- entropy annealing (`A -> B -> C`),
- `B-only` annealing,
- SAC-internal solidification/distillation variants:
  - `B-only baseline`
  - `B-only + solidify v1`
  - `B-only + solidify v2`

### Outcome
These variants produced a very consistent pattern:

- deterministic mean action became larger,
- det/stoch action ratio increased,
- but deterministic geometry did not break through the outer-level ceiling.

### Interpretation
This phase suggested that mean activation and geometric control depth were not the same problem. Making the mean larger did not automatically create a deeper deterministic controller.

## Phase 6. Reframe SAC as Teacher and Add a Separate Deterministic Student
Because SAC continued to behave like a useful explorer but a weak final deterministic controller, the stage was extended with a teacher-student path:

- build teacher datasets from stored artifacts,
- train a separate deterministic student by weighted regression to executed action,
- evaluate the student against teacher deterministic baselines.

### Teacher-Student v1
- dataset v1: `100` samples
- inner: `9`
- dwell: `0`

### Outcome
Student v1 matched outer-level teacher behavior but did not surpass it.

## Phase 7. Expand the Teacher Data Pool
Because the first student dataset was obviously shallow, a larger collection campaign was run:

- two additional B-only collection runs,
- two additional solidify collection runs,
- merged with prior teacher runs into `det_extract_v2`.

### Dataset v2 Outcome
- total: `456`
- elite: `316`
- strong: `140`
- inner: `29`
- dwell: `9`
- outer: `268`

This was a substantial improvement over v1, but the dataset was still dominated by outer-level samples.

## Phase 8. Train Student v2 on the Larger Dataset
Student v2 was trained on `det_extract_v2`.

### Outcome
- retention improved relative to student v1,
- but the student still did not exceed the best teacher deterministic baseline,
- outer remained at `0.2`,
- inner remained at `0.0`,
- success remained at `0.2`,
- regression remained at `0.8`.

### Interpretation
The larger dataset helped the student become somewhat cleaner and less lossy, but not deeper. The bottleneck shifted from "insufficient data volume" to "insufficient deterministic consolidatability of the deeper behavior."

## What Was Ruled Out Along the Way
By the end of the stage, the following had been substantially ruled out as primary explanations:

- target generation bug as the dominant remaining issue,
- safety correction as the dominant remaining issue,
- curriculum as the dominant remaining issue,
- broad instability as the dominant remaining issue,
- simply increasing mean action magnitude as a sufficient solution,
- simply increasing dataset size as a sufficient solution.

## What Partially Worked
Several directions did help, but only partially:

- safety/mismatch handling reduced unstable pathological behavior,
- reward shaping enabled consistent stochastic outer-level approach,
- evaluation infrastructure made the real bottleneck visible,
- teacher-student extraction improved retention somewhat.

None of these, however, produced stable deterministic inner-level control.

## Final Stage Interpretation
The V5.1 stage should be read as a narrowing process:

1. the project started with broad instability and multiple confounds,
2. those confounds were progressively reduced,
3. useful stochastic approach behavior was established,
4. but deterministic consolidation repeatedly stalled at the outer level.

## Final Concluding Statement
The V5.1 stage narrowed the problem from a broad system-instability regime to a much more specific control bottleneck: the stochastic teacher can reliably learn outer-level approach behavior, but deeper inner-level behavior still cannot be consistently consolidated into a deterministic controller.
