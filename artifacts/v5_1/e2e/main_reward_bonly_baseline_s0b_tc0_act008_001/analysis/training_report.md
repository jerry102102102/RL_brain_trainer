# Training Report: main_reward_bonly_baseline_s0b_tc0_act008_001

## Headline
- Episodes completed: 70
- Train success rate: 0.057
- Best min dpos: 0.0075
- Mean final dpos: 0.1297
- Regression rate: 0.900
- True basin hit rate: 0.543
- True outer / inner / dwell rates: 0.429 / 0.057 / 0.057
- True final basin rate: 0.171

## Deterministic Eval
- Episodes completed: 5
- Success rate: 0.200
- Mean final dpos: 0.0985
- Regression rate: 0.800

## Gap Diagnosis
- deterministic: success=0.200, basin=0.200, inner=0.000, dwell=0.000, final_dpos=0.0985, action_l2=0.0034, raw_norm=0.0427
- noise010: success=0.200, basin=0.200, inner=0.000, dwell=0.000, final_dpos=0.1070, action_l2=0.0183, raw_norm=0.2292
- noise030: success=0.200, basin=0.600, inner=0.000, dwell=0.000, final_dpos=0.1385, action_l2=0.0535, raw_norm=0.6685
- noise060: success=0.200, basin=0.200, inner=0.000, dwell=0.000, final_dpos=0.1668, action_l2=0.0874, raw_norm=1.0931

## Solidification
- Active distill lambda: 0.0000
- Distill good fraction: 0.0000
- Distill quality mean: 0.0000
- Distill mean/target action L2: 0.0000 / 0.0000
- Distill advantage mean: 0.0000

## Entropy Annealing
- Mode: event
- Current stage: B
- Current target entropy: -4.900
- Stage switches: 1
- ep20: A->B, target=-4.900, reason=event_train_basin=0.450_inner=0.100_reject=0.025

## Plots
- [Distance To Target](plots/distance_to_target.png)
- [Zone Success Rates](plots/zone_success_rates.png)
- [Action Safety](plots/action_safety.png)
- [Periodic Deterministic Eval](plots/periodic_deterministic_eval.png)
- [Entropy Annealing](plots/entropy_annealing.png)
- [Gap Noise Sweep](plots/gap_noise_sweep.png)
- [Solidification Metrics](plots/solidification_metrics.png)

## Observations
- Run produced 4 successful episodes.
- Most episodes ended farther from the target than their closest point.
- Post-train deterministic evaluation is available.
