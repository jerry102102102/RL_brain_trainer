# V3 Complexity-Escalation Ablation

## Setup
- Branch: `v3-online-memory`
- Script: `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`
- Base style: existing `three_layer_lstm_ablation` seed-aggregate report format
- Modes compared:
  - A: `core_only` (no middle-layer residual)
  - B: `core_plus_lstm_memory_residual` (LSTM memory residual + conditional gate)
- Fairness controls held identical inside each scenario: seeds, train/eval episodes, planner (L1), controller (L3), deterministic core mapping, optimizer/hparams, timeout bins.
- Only intervention: L2 residual path enabled in mode B.
- Seeds: `[11, 29, 47]` (3 seeds per setting)

## Harder Settings
- S1: hard disturbances (`disturbance_level=hard`), `obstacle_count=0`, `max_steps=220`
- S2: hard disturbances (`disturbance_level=hard`), `obstacle_count=4`, `max_steps=220`

## Concise Result Table (mean +/- std over seeds)

| scenario | mode | success_rate | tracking_rmse | control_effort | done_reasons (success/collision/timeout) | timeout bins near/mid/far |
|---|---|---:|---:|---:|---:|---:|
| S1 hard-disturbance | A core_only | 1.000 +/- 0.000 | 0.774 +/- 0.063 | 57.247 +/- 3.923 | 30.0 +/- 0.0/0.0 +/- 0.0/0.0 +/- 0.0 | 0.0 +/- 0.0/0.0 +/- 0.0/0.0 +/- 0.0 |
| S1 hard-disturbance | B core_plus_lstm_memory_residual | 0.933 +/- 0.027 | 0.723 +/- 0.050 | 65.633 +/- 4.325 | 28.0 +/- 0.8/0.0 +/- 0.0/2.0 +/- 0.8 | 1.7 +/- 0.5/0.3 +/- 0.5/0.0 +/- 0.0 |
| S2 hard+obstacles | A core_only | 0.856 +/- 0.016 | 0.870 +/- 0.053 | 51.848 +/- 2.680 | 25.7 +/- 0.5/4.3 +/- 0.5/0.0 +/- 0.0 | 0.0 +/- 0.0/0.0 +/- 0.0/0.0 +/- 0.0 |
| S2 hard+obstacles | B core_plus_lstm_memory_residual | 0.844 +/- 0.042 | 0.811 +/- 0.076 | 61.314 +/- 2.008 | 25.3 +/- 1.2/3.7 +/- 1.2/1.0 +/- 0.0 | 1.0 +/- 0.0/0.0 +/- 0.0/0.0 +/- 0.0 |

## Delta Summary (B - A)

| scenario | success_rate | tracking_rmse | control_effort | timeout_near | timeout_mid | timeout_far |
|---|---:|---:|---:|---:|---:|---:|
| S1 hard-disturbance | -0.067 | -0.051 | +8.387 | +1.7 | +0.3 | +0.0 |
| S2 hard+obstacles | -0.011 | -0.059 | +9.466 | +1.0 | +0.0 | +0.0 |

## Recommendation
- L2 is not yet beneficial by mission success in this escalation: B lowers tracking RMSE but reduces success_rate and increases control effort. Use A as default, and only enable B when smoother tracking is prioritized over success_rate.
- Practical trigger for enabling B: require non-negative success-rate delta on the target setting and acceptable control-effort increase.
