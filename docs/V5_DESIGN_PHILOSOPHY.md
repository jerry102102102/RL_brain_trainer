# Robot_brain_trainer v5 — Design Philosophy and System Direction

Status: Adopted (2026-03-10)
Scope: V5 architecture/training doctrine (not low-level implementation)

## 1) Purpose
This document defines **how we think about training and architecture in V5**:
- what kind of robot brain we are building,
- why we separate L1/L2/L3,
- why training should focus on L2,
- why simulation is first battlefield,
- what counts as real progress.

## 2) Core belief (write this in stone)
A strong robot system should **not** treat all intelligence as one monolithic policy.

We explicitly reject:
- `image + instruction -> end-to-end raw action`

because it is usually:
- hard to debug,
- hard to verify,
- hard to stabilize,
- hard to improve incrementally,
- hard to localize failure source.

Our central doctrine:
> **Separate understanding from skill, and skill from precision.**

## 3) Layer contract (L1 / L2 / L3)

### L1 — semantic / global intent
Answers:
- What is the task?
- What is the object?
- What is the target?
- What constraints matter?

Must **not** own precise motor behavior.

### L2 — local planning / policy (main learning locus)
Answers:
- Given current state + goal, what local skill action now?
- Which primitive/policy should be used now?
- How to react under local uncertainty?

This is where RL/shaping should focus.

### L3 — deterministic follower
Answers:
- How to execute stably?
- How to guarantee tracking/safety/smoothness/physical realism?

Must **not** need language/task semantics.

## 4) Training worldview (shaping-first)
Priority:
1. Train **L2 skill** first
2. Align **L1 -> skill selection/parameterization**
3. Tighten **L3 precision** (still deterministic-first)

Not recommended:
- train all layers jointly end-to-end and hope behavior emerges.

## 5) Simulation-first doctrine
Current phase target is **not real-world deployment**.
Current phase target is:
- structurally correct,
- trainable,
- diagnosable,
- repeatable in simulation.

Simulation must prove:
1. Architecture correctness (L1/L2/L3 cooperate as designed)
2. Training correctness (L2 actually learns useful skills)
3. Evaluation correctness (clear pass/fail/generalization metrics)
4. Reproducible loop (rerun/compare/improve experiments)

## 6) VLM/VLA usage rule
VLM is for:
- intent interpretation,
- grounding,
- structured command generation,
- disambiguation/verification support.

VLM is **not** raw motor policy.
Do not collapse L1/L2/L3 boundaries by placing VLM at execution layer.

## 7) Practical staged pipeline
- Stage A: workspace intelligence (safe movement priors)
- Stage B: atomic skills (approach, align, grasp, lift, transfer, place, retreat)
- Stage C: composed task behavior (multi-step stability + recovery)
- Stage D: semantic alignment (L1 maps to stable L2 skill graph)

## 8) Progress criteria
Good progress means:
- clearer layer responsibility,
- better L2 trainability,
- easier failure attribution (L1 vs L2 vs L3),
- reusable/composable skills,
- robustness to pose/layout variation,
- verifiable and reproducible results.

## 9) Design decision checklist
Before accepting a design:
1. Does it preserve L1/L2/L3 separation?
2. Does it improve L2 trainability?
3. Can it be cleanly evaluated in sim?
4. If it fails, can we localize the failing layer?
5. Does it improve reusable skill building (not single-case hacks)?

If most answers are yes -> aligned with V5 direction.

## 10) Implementation policy note for this repo
- L3 remains deterministic-first.
- L2 is primary learning surface.
- WP-level execution should prefer shaping/curriculum over blind trial-and-error.
- New experiments must include layer-boundary diagnostics.

---
Source basis: user-provided "Robot_brain_trainer v5 Design Philosophy and System Direction" (PDF, 2026-03-10).
