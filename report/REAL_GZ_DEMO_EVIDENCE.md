# Real Gazebo Demo Evidence

Purpose: document the non-slide demo evidence for the final package. This file is intentionally separate from the report-style generated videos: it records the actual Gazebo / ROS2 run where the robot arm was controlled through the Phase 3A `Approach -> Finisher` runtime path.

## What This Demonstrates

This is the closest current demo to the requested real simulation evidence:

```text
Gazebo kitchen scene
-> ROS2 /joint_states
-> frozen Approach policy
-> frozen Finisher policy
-> /arm_controller/follow_joint_trajectory action path
-> Gazebo camera video
```

It is not a full holder1 -> holder8 tray transport solution. It is a controlled Phase 3A simulation validation showing that the trained RL policy stack can issue trajectory goals to the Gazebo arm controller and produce real motion in the scene.

## Verified Run

- Run id: `final_real_gz_sensor_demo_005`
- Camera video: `report/videos/real_gz_camera_phase3a_controlled_sim.mp4`
- Extracted preview frame: `report/video_frames/real_gz_camera_phase3a_frame_5s.png`
- Runtime summary: `artifacts/v5/phase3a_controlled_sim/final_real_gz_sensor_demo_005/controlled_sim_summary.json`
- Step log: `artifacts/v5/phase3a_controlled_sim/final_real_gz_sensor_demo_005/runtime_steps.jsonl`
- Camera summary: `report/demo_outputs/demo_04_real_gz_camera_summary.json`
- Compact runtime summary: `report/demo_outputs/demo_04_real_gz_controlled_sim_summary.json`

## Actual Results

The verified run used three `visible_workspace` FK targets.

| Metric | Value |
|---|---:|
| Target count | 3 |
| Runtime steps | 60 |
| ROS execution OK count | 60 / 60 |
| Command path | `action` |
| Success rate | 0.3333 |
| Mean final position error | 0.04586 m |
| Mean final orientation error | 0.03366 rad |
| Mean actual joint delta L2 | 0.01484 |
| Max actual joint delta L2 | 0.05152 |
| Mean tracking error L2 | 0.0000807 |
| Camera frames captured | 251 |
| Camera video resolution | 640 x 480 |
| Camera video duration | 25.1 s |

## Interpretation

This result is useful but limited:

- Confirmed: Gazebo scene launches and publishes camera / joint-state topics.
- Confirmed: the runtime path sends commands through the ROS action interface, not through a fake animation.
- Confirmed: all 60 controller goals in this run executed successfully.
- Confirmed: a Gazebo camera MP4 was captured from `/v5/cam/side/rgb`.
- Limitation: this visible-workspace stress run only solved 1 of 3 targets under the current strict metric.
- Limitation: this is not a successful full tray transport demo.

The honest conclusion is that the final package now has real Gazebo execution evidence, but the strongest quantitative result remains the kinematic `Approach -> Finisher` skill stack and the route-curriculum prefix extension.

## Rerun Command

Use the existing verified artifact:

```bash
bash scripts/final/run_demo_04_real_gz_sensor_demo.sh
```

Rerun Gazebo and regenerate the camera video:

```bash
RERUN_REAL_GZ_DEMO=1 RUN_ID=final_real_gz_sensor_demo_rerun_001 \
  bash scripts/final/run_demo_04_real_gz_sensor_demo.sh
```

The rerun command starts the kitchen scene, waits for `/joint_states` and `/v5/cam/side/rgb`, records the Gazebo camera topic, runs the RL controlled sim, and writes both video and runtime logs.
