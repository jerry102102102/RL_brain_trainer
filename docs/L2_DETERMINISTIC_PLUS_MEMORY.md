# L2 Deterministic Core + Memory Residual (Fixed L1/L3, No LSTM)

## Setup
- Branch: `v3-online-memory`
- Script: `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`
- Config mode: `l2_deterministic_plus_memory`
- Modes: `core_only` vs `core_plus_memory_residual`
- Deterministic core mapping: heading+distance to desired `[v, omega]`
- Memory residual clip: `+/-0.15`
- Obstacles forced to `0` for train/eval

## Results

| Metric | core_only | core_plus_memory_residual | Delta (plus-core) |
|---|---:|---:|---:|
| success_rate | 1.000 | 0.700 | -0.300 |
| avg_return | -45.958 | -70.653 | -24.695 |
| done_reasons.success | 40 | 28 | -12 |
| done_reasons.timeout | 0 | 12 | +12 |
| timeout_distance_bins.near | 0 | 10 | +10 |
| timeout_distance_bins.mid | 0 | 2 | +2 |
| timeout_distance_bins.far | 0 | 0 | +0 |

## Done Reasons
- core_only: `{"success": 40}`
- core_plus_memory_residual: `{"timeout": 12, "success": 28}`
