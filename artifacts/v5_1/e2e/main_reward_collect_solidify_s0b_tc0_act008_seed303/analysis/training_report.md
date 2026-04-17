# Training Report: main_reward_collect_solidify_s0b_tc0_act008_seed303

## Headline
- Episodes completed: 100
- Train success rate: 0.000
- Best min dpos: 0.0237
- Mean final dpos: 0.1300
- Regression rate: 0.920
- True basin hit rate: 0.420
- True outer / inner / dwell rates: 0.350 / 0.060 / 0.010
- True final basin rate: 0.100

## Deterministic Eval
- Episodes completed: 3
- Success rate: 0.667
- Mean final dpos: 0.0877
- Regression rate: 0.333

## Gap Diagnosis
- deterministic: success=0.667, basin=0.667, inner=0.000, dwell=0.000, final_dpos=0.0877, action_l2=0.0055, raw_norm=0.0693
- noise030: success=0.000, basin=0.333, inner=0.000, dwell=0.000, final_dpos=0.1369, action_l2=0.0490, raw_norm=0.6120
- noise060: success=0.000, basin=0.667, inner=0.000, dwell=0.333, final_dpos=0.1518, action_l2=0.0870, raw_norm=1.0871

## Solidification
- Active distill lambda: 0.0500
- Distill good fraction: 0.0000
- Distill quality mean: 0.0000
- Distill mean/target action L2: 0.0000 / 0.0000
- Distill advantage mean: 0.0000

## Entropy Annealing
- Mode: event
- Current stage: B
- Current target entropy: -4.900
- Stage switches: 1
- ep40: A->B, target=-4.900, reason=event_train_basin=0.400_inner=0.100_reject=0.010

## Plots
- [Distance To Target](plots/distance_to_target.png)
- [Zone Success Rates](plots/zone_success_rates.png)
- [Action Safety](plots/action_safety.png)
- [Periodic Deterministic Eval](plots/periodic_deterministic_eval.png)
- [Entropy Annealing](plots/entropy_annealing.png)
- [Gap Noise Sweep](plots/gap_noise_sweep.png)
- [Solidification Metrics](plots/solidification_metrics.png)

## Observations
- No successful training episodes were recorded.
- Most episodes ended farther from the target than their closest point.
- Post-train deterministic evaluation is available.
