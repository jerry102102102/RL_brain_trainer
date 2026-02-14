# V3 Hierarchy Meaning Ablation (Template)

## Goal
Validate L2 semantics under the contract:
- L1 emits semantic/regional planning representation only (not direct control).
- L2 plans local route/trajectory.
- L3 is a fixed deterministic follower.

## Experiment Spec
- Branch: `v3-online-memory`
- Runner:
  - `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/run_v3_hierarchy_meaning_ablation.py`
  - or `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py` with `train_mode: hierarchy_meaning_ablation`
- Config:
  - `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_hierarchy_meaning_ablation.yaml`
- Seeds: `[11, 29, 47]`

## Difficulty Tiers
- `low`: fewer obstacles, lower route complexity, homogeneous easy disturbance.
- `medium`: moderate obstacles, moderate route complexity, mixed easy/medium disturbances.
- `high`: dense obstacles, high route complexity, heterogeneous easy/medium/hard disturbances.

## Compared Modes
- A: `no_l2_shortcut`
  - L1 semantic target passed directly to L3 (bypasses local L2 trajectory planning).
- B: `l2_no_memory`
  - L2 local trajectory planner enabled, without memory/LSTM.
- C: `l2_memory_lstm`
  - L2 local trajectory planner with memory residual and LSTM residual.

## Metrics (mean/std over seeds)
- `success_rate`
- `done_reasons`
- `timeout_distance_bins` (`near`, `mid`, `far`)
- Trajectory quality:
  - `path_efficiency`
  - `progress_ratio`
  - `waypoint_tracking_rmse`
  - `min_obstacle_clearance`
- `control_effort`

## Run Command
```bash
cd hrl_ws
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=src/hrl_trainer \
  uv run python -m hrl_trainer.sim2d.run_v3_hierarchy_meaning_ablation \
  --config src/hrl_trainer/config/train_rl_brainer_v3_hierarchy_meaning_ablation.yaml \
  --out /tmp/v3_hierarchy_meaning_ablation.json
```

## Result Matrix (Fill After Run)

| tier | mode | success_rate (mean +/- std) | done_reasons (mean counts) | timeout near/mid/far (mean counts) | path_efficiency | waypoint_tracking_rmse | control_effort |
|---|---|---:|---|---:|---:|---:|---:|
| low | A no_l2_shortcut | TODO | TODO | TODO | TODO | TODO | TODO |
| low | B l2_no_memory | TODO | TODO | TODO | TODO | TODO | TODO |
| low | C l2_memory_lstm | TODO | TODO | TODO | TODO | TODO | TODO |
| medium | A no_l2_shortcut | TODO | TODO | TODO | TODO | TODO | TODO |
| medium | B l2_no_memory | TODO | TODO | TODO | TODO | TODO | TODO |
| medium | C l2_memory_lstm | TODO | TODO | TODO | TODO | TODO | TODO |
| high | A no_l2_shortcut | TODO | TODO | TODO | TODO | TODO | TODO |
| high | B l2_no_memory | TODO | TODO | TODO | TODO | TODO | TODO |
| high | C l2_memory_lstm | TODO | TODO | TODO | TODO | TODO | TODO |

## Interpretation Prompts
- Does B outperform A as route complexity and obstacle density rise?
- Does C outperform B under disturbance heterogeneity (especially high tier)?
- Which timeout bins change most across A/B/C?
- How do trajectory quality and control effort trade off across modes?
