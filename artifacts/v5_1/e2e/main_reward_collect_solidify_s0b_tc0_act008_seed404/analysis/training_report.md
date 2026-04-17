# Training Report: main_reward_collect_solidify_s0b_tc0_act008_seed404

## Headline
- Episodes completed: 120
- Train success rate: 0.092
- Best min dpos: 0.0142
- Mean final dpos: 0.1240
- Regression rate: 0.900
- True basin hit rate: 0.525
- True outer / inner / dwell rates: 0.450 / 0.033 / 0.042
- True final basin rate: 0.208

## Deterministic Eval
- Episodes completed: 3
- Success rate: 0.667
- Mean final dpos: 0.0780
- Regression rate: 0.333

## Gap Diagnosis
- deterministic: success=0.667, basin=0.667, inner=0.000, dwell=0.000, final_dpos=0.0780, action_l2=0.0065, raw_norm=0.0808
- noise030: success=0.000, basin=0.333, inner=0.000, dwell=0.000, final_dpos=0.1363, action_l2=0.0503, raw_norm=0.6289
- noise060: success=0.000, basin=0.667, inner=0.000, dwell=0.333, final_dpos=0.1224, action_l2=0.0850, raw_norm=1.0619

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
- ep40: A->B, target=-4.900, reason=event_train_basin=0.750_inner=0.100_reject=0.005

## Plots
- [Distance To Target](plots/distance_to_target.png)
- [Zone Success Rates](plots/zone_success_rates.png)
- [Action Safety](plots/action_safety.png)
- [Periodic Deterministic Eval](plots/periodic_deterministic_eval.png)
- [Entropy Annealing](plots/entropy_annealing.png)
- [Gap Noise Sweep](plots/gap_noise_sweep.png)
- [Solidification Metrics](plots/solidification_metrics.png)

## Observations
- Run produced 11 successful episodes.
- Most episodes ended farther from the target than their closest point.
- Post-train deterministic evaluation is available.
