# Phase 3A RL Scene Route Check

Purpose: summarize the first Gazebo headless check that used the trained RL `Approach -> Finisher` models, not IK commands, against the real kitchen-scene tray route from holder 1 to holder 8.

## Setup

- Scene source: `external/kitchen_scene`
- Route geometry source: external `MoveTaskLibrary` tray/holder poses only
- Controller under test: trained RL `Approach -> Finisher`
- IK usage: none for the RL execution path
- Run artifact: `artifacts/v5/phase3a_controlled_sim/method1_rl_scene_pose_tray1_h1_to_h8_001`
- Per-waypoint analysis: `artifacts/v5/phase3a_controlled_sim/method1_rl_scene_pose_tray1_h1_to_h8_001/scene_pose_rl_partial_analysis.json`

## Route Scale

The scene-level tray route is much larger than the local visible-workspace targets used by the current trained RL policy.

| Index | Waypoint | Approx. target position | Delta from previous |
|---:|---|---|---:|
| 0 | `home_src_orientation` | `[-0.180, 0.002, 1.100]` | n/a |
| 1 | `src_pre_approach` | `[-0.920, -0.381, 0.407]` | `1.084 m` |
| 2 | `src_approach` | `[-0.920, -0.761, 0.407]` | `0.380 m` |
| 4 | `src_lifted_clear` | `[-0.920, -0.761, 0.557]` | `0.150 m` |
| 5 | `src_retract_sky` | `[-0.920, -0.211, 0.600]` | `0.552 m` |
| 6 | `mid_carry_arc` | `[-0.709, 0.000, 0.600]` | `0.298 m` |
| 7 | `dst_side_sky` | `[-0.920, 0.211, 0.600]` | `0.298 m` |
| 8 | `dst_pre_approach` | `[-0.920, 0.681, 0.507]` | `0.479 m` |
| 13 | `home_done` | `[-0.180, 0.002, 1.100]` | `0.903 m` |

The route also includes large yaw changes, including the destination side at approximately `pi` radians.

## RL Result

The run was stopped after enough evidence was collected. The runtime was actively publishing action-primary RL commands, and tracking error to commanded joint positions stayed small, so this was not a ROS controller inactivity issue.

| Target | Strict success | Handoff ready | Min pos err | Final pos err | Min ori err | Final ori err |
|---|---:|---:|---:|---:|---:|---:|
| `home_src_orientation` | false | false | `0.0068 m` | `0.0245 m` | `0.1046 rad` | `0.1271 rad` |
| `src_pre_approach` | false | false | `1.0746 m` | `1.1149 m` | `0.1007 rad` | `1.1344 rad` |
| `src_approach` | false | false | `0.7177 m` | `0.8751 m` | `1.1466 rad` | `2.3736 rad` |
| `src_insert_under_tray` | false | false | `0.7136 m` | `0.7149 m` | `2.3275 rad` | `3.0941 rad` |
| `src_lifted_clear` | false | false | `0.7641 m` | `0.7923 m` | `3.0988 rad` | `3.2014 rad` |
| `src_retract_sky` | false | false | `0.4318 m` | `0.4446 m` | `3.1470 rad` | `3.1478 rad` |
| `mid_carry_arc` | false | false | `0.3593 m` | `0.6105 m` | `0.0657 rad` | `0.7783 rad` |
| `dst_side_sky` | false | false | `0.3011 m` | `0.3457 m` | `1.5557 rad` | `3.2832 rad` |
| `dst_pre_approach` | false | false | `0.0642 m` | `0.1458 m` | `3.1883 rad` | `3.2560 rad` |

## Interpretation

The current RL policy is working as a local learned controller, but it has not been trained for the full holder-to-holder tray route. The full scene task requires large absolute workspace motion, large changes in height, and large yaw changes. The current trained models were validated on local FK/visible-workspace targets, not on source-to-destination rack-level transport.

The old IK route can be useful as a diagnostic or teacher source, but it should not be treated as the project controller result. The project controller result should remain the trained RL model. Under that constraint, the current evidence says the trained RL model cannot yet perform the full tray1 holder1-to-holder8 route.

## Current Conclusion

The Phase 3A integration path is live enough to run RL commands in Gazebo, but the trained `Approach -> Finisher` policy does not yet cover the real scene-level tray transport workspace. For the final demo, the honest RL-backed result should be scoped to local approach / pose-hold behavior, unless a new large-workspace RL training pass is completed.
