# V3 Level-5 Pentagon Ablation

## Setup
- Branch: `v3-online-memory`
- Runner: `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/run_v3_hierarchy_meaning_ablation.py`
- Config: `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_level5_pentagon_ablation.yaml`
- Output JSON: `/tmp/v3_level5_pentagon_ablation.json`
- Seeds: `[11, 29, 47, 83, 131]` (5 seeds)
- Modes:
  - A: `no_l2_shortcut`
  - B: `l2_no_memory`
  - C: `l2_memory_lstm`

## Hierarchy Contract
- L1: semantic/regional route packet only (`semantic_intent`, `region_waypoints`, route tags, handoff radius, speed hint).
- L2: local trajectory planner from semantic packet.
- L3: fixed deterministic follower (`_l3_follow_trajectory` + fixed RBF low-level controller).

## Pentagon Collision / Failure Rule
Implemented in `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/env.py`:
- Robot footprint treated as regular pentagon using circumscribed radius approximation:
  - `robot_apothem = 0.09`
  - `robot_circ_radius = apothem / cos(pi/5)`
- Obstacle contact: center-distance `<= obstacle_radius + robot_circ_radius`.
- Wall contact: `abs(x)` or `abs(y)` reaches `world_half_extent - robot_circ_radius`.
- Any obstacle or wall contact triggers failure termination (`collided=True`, `done=True`, `success=False`).

## Fairness Controls
- Same seeds for all A/B/C per tier.
- Same environment dynamics, disturbances, obstacle counts, episode limits, and evaluation episodes across A/B/C within each tier.
- Same L3 fixed deterministic follower and low-level controller across all modes.
- Only L2 capability differs by mode:
  - A bypasses L2 trajectory generation.
  - B uses deterministic L2 trajectory planner only.
  - C uses same L2 planner plus memory+LSTM residual correction.

## Tier Design (5 Levels)
- `level1`: very low route complexity, 2 obstacles, easy disturbance.
- `level2`: low route complexity, 4 obstacles, easy/medium disturbances.
- `level3`: medium route complexity, 6 obstacles, easy/medium disturbances.
- `level4`: high route complexity, 8 obstacles, easy/medium/hard disturbances.
- `level5`: extreme route complexity, 10 obstacles, medium/hard disturbances.

## Key Results (mean +/- std over 5 seeds)

| Tier | A success | B success | C success | Delta B-A | Delta C-B | Delta C-A |
|---|---:|---:|---:|---:|---:|---:|
| level1 | 0.624 +/- 0.060 | 0.928 +/- 0.039 | 0.928 +/- 0.059 | +0.304 | +0.000 | +0.304 |
| level2 | 0.520 +/- 0.050 | 0.807 +/- 0.106 | 0.787 +/- 0.113 | +0.287 | -0.020 | +0.267 |
| level3 | 0.349 +/- 0.052 | 0.560 +/- 0.043 | 0.531 +/- 0.046 | +0.211 | -0.029 | +0.183 |
| level4 | 0.230 +/- 0.029 | 0.330 +/- 0.075 | 0.350 +/- 0.077 | +0.100 | +0.020 | +0.120 |
| level5 | 0.151 +/- 0.065 | 0.142 +/- 0.030 | 0.147 +/- 0.056 | -0.009 | +0.004 | -0.004 |

| Tier | Mode | Return mean | Path efficiency | Waypoint RMSE | Control effort |
|---|---|---:|---:|---:|---:|
| level1 | A | -38.296 | 0.667 | 0.628 | 34.180 |
| level1 | B | -64.242 | 0.727 | 0.111 | 31.725 |
| level1 | C | -57.628 | 0.749 | 0.152 | 29.446 |
| level2 | A | -50.460 | 0.601 | 0.594 | 40.646 |
| level2 | B | -85.325 | 0.573 | 0.109 | 55.988 |
| level2 | C | -77.041 | 0.594 | 0.146 | 50.937 |
| level3 | A | -49.613 | 0.658 | 0.692 | 37.268 |
| level3 | B | -107.973 | 0.560 | 0.114 | 72.014 |
| level3 | C | -98.918 | 0.576 | 0.149 | 65.558 |
| level4 | A | -48.437 | 0.586 | 0.847 | 34.932 |
| level4 | B | -131.232 | 0.440 | 0.120 | 78.081 |
| level4 | C | -120.749 | 0.459 | 0.151 | 72.760 |
| level5 | A | -49.085 | 0.592 | 0.991 | 34.440 |
| level5 | B | -163.545 | 0.340 | 0.125 | 78.938 |
| level5 | C | -156.303 | 0.355 | 0.151 | 76.240 |

## Conclusion: L2 Importance Progression by Difficulty
- L2 trajectory planning (B vs A) is strongly beneficial from level1 to level4 on success rate, with the largest gains in level1-3 and still positive in level4.
- At level5 (extreme), B no longer outperforms A on success (slight negative delta), indicating this setup is near/at failure regime for current L2 planner under strict collision termination.
- Memory+LSTM residual (C vs B) gives mixed gains:
  - Neutral to negative in level1-3.
  - Positive in level4 and marginally positive in level5.
- Net interpretation:
  - L2 remains important through moderate/high difficulty but saturates in extreme difficulty under pentagon contact failure constraints.
  - C-mode memory/LSTM helps most near the high-difficulty boundary (level4+) but is not sufficient alone to recover strong success in level5.
