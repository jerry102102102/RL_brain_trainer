# L2 Memory Isolation Ablation (Fixed L1/L3, No LSTM)

## Setup
- Branch: `v3-online-memory`
- Script: `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`
- Config: `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_l2_ablation_quick.yaml`
- Run command:
  - `cd hrl_ws && UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=src/hrl_trainer uv run python -m hrl_trainer.sim2d.train_rl_brainer_v3_online --config src/hrl_trainer/config/train_rl_brainer_v3_l2_ablation_quick.yaml --out /tmp/l2_memory_ablation.json`

## Controlled Conditions
- High-level planner fixed: `HighLevelHeuristicPlannerV2`
- Low-level controller fixed: `_rbf_controller`
- Tactical policy: non-recurrent feed-forward baseline (no LSTM)
- `obstacle_count` forced to `0` in training and eval environments
- Same seed/config for both modes: `seed=11`
- Only ablated factor: episodic memory usage
  - `memory_off`: no retrieval (memory feature zeros), no memory bank accumulation
  - `memory_on`: nearest-neighbor episodic retrieval enabled with accumulation

## Results (Eval: 40 episodes each)

| Metric | memory_off | memory_on | Delta (on-off) |
|---|---:|---:|---:|
| success_rate | 0.075 | 0.125 | +0.050 |
| done_reasons.success | 3 | 5 | +2 |
| done_reasons.timeout | 37 | 35 | -2 |
| timeout_distance_bins.near | 0 | 0 | 0 |
| timeout_distance_bins.mid | 2 | 2 | 0 |
| timeout_distance_bins.far | 35 | 33 | -2 |

## Conclusion
Under fixed upper/lower layers and a no-LSTM tactical model, memory accumulation/retrieval provided a modest improvement:
- Higher success rate (`+5` percentage points).
- Fewer timeout failures overall (`-2`) and fewer far-distance timeouts (`-2`).

In this run, memory helped, but the effect size is small and should be validated with multiple seeds before claiming robust gains.
