# V3 Three-Layer LSTM Ablation

## Setup
- Branch: `v3-online-memory`
- L1 planner fixed: `HighLevelHeuristicPlannerV2`
- L3 controller fixed: `rbf_controller`
- L2 deterministic core fixed: heading+distance mapping to desired `[v, omega]`
- Ablations:
  - A: `core_only`
  - B: `core_plus_memory_residual`
  - C: `core_plus_lstm_memory_residual` (LSTM residual over deterministic core, bounded and gated)
- Seeds: `[11, 29, 47]` (count=3)
- Fairness controls: same seeds/env/eval episodes per mode, obstacle_count forced to `0`, identical L1/L3 and core mapping

## Key Hyperparameters
- train_episodes: `70`
- eval_episodes: `30`
- seq_len: `10`
- hidden_dim: `128`
- memory_residual_clip: `0.12`
- lstm_residual_clip: `0.1`
- gate: `{"enabled": true, "near_goal_threshold": 0.3, "uncertainty_threshold": 0.06}`

## Aggregate Results (mean +/- std over seeds)

| mode | success_rate | avg_return | tracking_rmse | control_effort | timeout near/mid/far |
|---|---:|---:|---:|---:|---:|
| A core_only | 0.856 +/- 0.016 | -48.301 +/- 4.226 | 0.870 +/- 0.053 | 51.848 +/- 2.680 | 0.0/0.0/0.0 |
| B core+memory | 0.811 +/- 0.016 | -61.283 +/- 4.096 | 0.809 +/- 0.068 | 65.416 +/- 1.895 | 1.7/0.3/0.0 |
| C core+LSTM+memory | 0.844 +/- 0.042 | -57.966 +/- 3.772 | 0.811 +/- 0.076 | 61.314 +/- 2.008 | 1.0/0.0/0.0 |

## Conclusions
- Delta B-A success_rate: -0.044
- Delta C-B success_rate: +0.033
- Delta C-A success_rate: -0.011
- Delta C-A avg_return: -9.665
- LSTM gate activation (C): 0.364 +/- 0.023
