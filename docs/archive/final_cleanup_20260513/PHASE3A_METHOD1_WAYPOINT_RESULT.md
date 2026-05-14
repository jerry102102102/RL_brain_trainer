# Phase 3A Method 1 Waypoint Trial

This note records the first concrete test of the "VLM/LLM outputs waypoints, L2/L3 executes them" direction.

## Goal

Test whether the current Phase 3A stack can execute a tray-carry-like waypoint sequence before full image-grounded VLM perception is ready.

The intended contract is:

- L1/VLM chooses semantic waypoints.
- L1 does not output joint trajectories or raw actions.
- L2/L3 execute the waypoints using the trained `Approach -> Finisher` policies and existing ROS/Gazebo runtime boundary.

## External IK Reference

The existing external kitchen scene repo contains a tray waypoint/IK route:

- `external/kitchen_scene/src/kitchen_robot_controller/kitchen_robot_controller/task_library.py`
- `external/kitchen_scene/src/kitchen_robot_controller/kitchen_robot_controller/task_executor_node.py`

The old flow generates holder-to-holder tray poses with `MoveTaskLibrary`, solves IK for each pose, and publishes a `JointTrajectory` to `/arm_controller/joint_trajectory`.

Trial artifact:

- `artifacts/v5/method1_waypoint_ik/ik_tray_1_to_8_001/holder_task_executor.log`

Observed result:

- IK and waypoint generation succeeded.
- The old executor prepared a 124-point tray trajectory for `tray` holder `1 -> 8`.
- A publish probe confirmed a trajectory message appeared on `/arm_controller/joint_trajectory`.
- The simulated joint state did not move under this old topic-only path.

Interpretation:

- The old IK planner is useful as a waypoint design reference.
- The old topic-only executor is not the current reliable execution path.
- The current Phase 3A runtime should continue using the action-primary L3 path that already executed RL targets in Gazebo.

## VLM-Style RL Waypoint Plan

A new lightweight planner was added:

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/tray_waypoint_plan.py`

It creates two outputs:

- `vlm_tray_waypoint_plan.json`: human-readable L1-style semantic plan.
- `controlled_targets.json`: Phase 3A controlled-sim target list.

Generated plan artifact:

- `artifacts/v5/method1_waypoint_rl/vlm_tray1_a1_to_b1_001/vlm_tray_waypoint_plan.json`
- `artifacts/v5/method1_waypoint_rl/vlm_tray1_a1_to_b1_001/controlled_targets.json`

The first executable plan contains six semantic waypoints:

1. `pre_grasp_align`
2. `under_tray_insert_pose`
3. `level_lift`
4. `carry_midline`
5. `pre_insert_align`
6. `stable_insert_hold`

These are encoded as local FK-reachable `q_delta` targets so the demo remains inside the currently trained RL workspace.

## Gazebo Execution Result

Run artifact:

- `artifacts/v5/phase3a_controlled_sim/method1_vlm_waypoint_rl_tray1_a1_to_b1_006/controlled_sim_summary.json`
- `artifacts/v5/phase3a_controlled_sim/method1_vlm_waypoint_rl_tray1_a1_to_b1_006/runtime_steps.jsonl`
- `artifacts/v5/phase3a_controlled_sim/method1_vlm_waypoint_rl_tray1_a1_to_b1_006/waypoint_result_breakdown.json`

Result:

- Waypoints executed: `6`
- Success rate: `1.0`
- Handoff confirmed rate: `0.8333`
- Mean final position error: `0.00158 m`
- Mean final orientation error: `0.00985 rad`

This confirms that the current system can execute a VLM-style waypoint sequence through:

`semantic waypoint plan -> Phase 3A targets -> Approach -> Finisher -> ROS/Gazebo L3 execution`

## Current Limitation

This is not yet a full tray manipulation solution:

- The waypoint plan is semantic/mock local-FK, not image-grounded.
- The path does not yet solve contact-rich tray lifting.
- The external holder-to-holder IK planner uses a different table/world-frame trajectory and currently does not execute through the modern action-primary runtime.

## Current Conclusion

Method 1 is viable as a near-term demo path if scoped correctly:

> Let the VLM/LLM output structured semantic waypoints, then let the trained `Approach -> Finisher` runtime execute each waypoint through the existing safe L2/L3 path.

The next engineering step is to replace the local mock waypoint targets with frame-calibrated scene waypoints once the scene/object frames are reconciled.
