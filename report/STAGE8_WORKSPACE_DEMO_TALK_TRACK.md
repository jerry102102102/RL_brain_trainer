# Stage 8 Workspace Expansion Demo Talk Track

## Demo Claim

This demo is not claiming full kitchen transport is solved. The claim is narrower and more defensible:

> The learned `Approach -> Finisher` skill was expanded from a local kinematic controller into a larger workspace controller, with Stage 8 providing a visible expanded-workspace demonstration target.

## Why Stage 8

Stage 5 was the previous reliable trained workspace baseline:

- FK target envelope: about `20.4 cm x 59.7 cm x 4.9 cm`
- Prior Stage 5 success: about `0.93`
- This was reliable but visually still a local workspace.

Stage 8 is a larger curriculum stage:

- FK target envelope: about `35.3 cm x 85.1 cm x 10.7 cm`
- Joint target noise max: `+-0.330 rad`
- Joint target noise L2 radius: `0.544 rad`
- Latest 100-episode eval: `0.65` success, `0.78` finisher-ready rate
- Main failure mode: position error / timeout-regression, not catastrophic loss of all skill

This makes Stage 8 a good demo stage because it is clearly larger than Stage 5, but not so far outside the learned distribution that it becomes only a failure demo.

## How To Explain The Stage Table

Use this sentence:

> Each stage increases the joint-space target distribution and the resulting FK workspace envelope. Stage 0-5 preserve the original local skill, Stage 6-7 are now reliable expansion stages, and Stage 8 is the current visible expanded-workspace demo region. Stage 9-11 are stress tests that show where the next training bottleneck begins.

## Key Result Table

| Stage | FK Target Envelope | Latest Success | Interpretation |
|---:|---|---:|---|
| 5 | `20.4 x 59.7 x 4.9 cm` | `0.98` in latest run | Original reliable baseline retained |
| 6 | `24.7 x 67.7 x 6.2 cm` | `0.93` | Reliable expansion |
| 7 | `28.5 x 74.1 x 7.8 cm` | `0.83` | Reliable enough for expanded control claim |
| 8 | `35.3 x 85.1 x 10.7 cm` | `0.65` | Best current visible large-workspace demo stage |
| 9 | `48.5 x 100.2 x 15.0 cm` | `0.45` | Stress region, position bottleneck |
| 10 | `53.6 x 109.7 x 18.7 cm` | `0.40` | Stress region |
| 11 | `64.4 x 117.1 x 22.1 cm` | `0.29` | Outer stress region, not solved |

## Suggested Narration

> The original local skill was very accurate, but it was trained in a limited local workspace. To make it useful for a larger manipulation scene, I added workspace expansion curriculum stages. The important part is that we do not trust the latest model blindly. Every checkpoint is evaluated by deterministic stage gates, and the best checkpoint must preserve earlier stages while expanding later ones.

> Stage 8 is the current demonstration stage. It is substantially larger than the original Stage 5 workspace: roughly 35 cm by 85 cm by 11 cm in FK target envelope. The policy does not solve the full stress workspace yet, but it can now operate beyond the original local region. The remaining failures are mostly position error and timeout/regression at the larger stages, which points to the next research step: more late-stage fine-tuning and better anti-regression on long-distance targets.

## Honest Limitation Statement

Use this if asked whether the full scene task is solved:

> Not yet. The current system demonstrates modular L1-to-RL control, reliable local kinematic manipulation, and measurable workspace expansion. Full holder-to-holder Gazebo transport remains future work because the larger workspace stages still show position failures beyond Stage 8.

## Current Follow-Up Training

A late-stage fine-tuning run is being launched with:

- Initial checkpoint: `workspace_expand_stage10_11_ppo_1h_002/best_checkpoint/model_best_by_gate.zip`
- Start curriculum stage: `8`
- Best-checkpoint scoring stage: `8`
- More sampling weight on Stage 8-11
- Reduced old-stage replay, but not zero, to avoid catastrophic forgetting

The goal is to improve the Stage 8 demo region first, then see whether Stage 9-11 start becoming usable.
