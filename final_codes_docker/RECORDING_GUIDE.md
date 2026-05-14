# Recording Guide

This guide is for a 1-2 minute final code screen recording.

## Recommended Layout

- Left side: terminal with large font.
- Right side: Gazebo or RViz.
- Keep VS Code closed or minimized unless showing files is required.

## Segment 1: Local Test Demo

Command:

```bash
bash scripts/final/run_final_local_test_demo.sh
```

What to say:

> This demo runs the final local Approach -> Finisher skill. The headless path runs deterministic policy evaluation, while the native visual path can drive the Gazebo/RViz demo runtime.

Expected output:

```text
LOCAL TEST DEMO FINISHED
Outputs: report/demo_outputs/final_local_test_...
```

## Segment 2: Full Route / Route-Prefix Demo

Command:

```bash
bash scripts/final/run_final_full_route_demo.sh
```

What to say:

> This demo shows route-curriculum evidence. It evaluates dense route-prefix following and reports longest successful prefix. It does not claim the full holder1-to-holder8 transport task is solved.

Expected output:

```text
FULL ROUTE / ROUTE-PREFIX DEMO FINISHED
This is route-curriculum evidence, not full transport completion.
Outputs: report/demo_outputs/final_full_route_...
```

The packaged Docker/headless command defaults to the first 90 dense waypoints for CPU-stable reproduction. For the longer native prefix120 reference check, run:

```bash
FULL_ROUTE_END_INDEX=120 bash scripts/final/run_final_full_route_demo.sh
```

## Optional Native Gazebo Visual Recording

Terminal 1:

```bash
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
```

Terminal 2:

```bash
rviz2 -d config/rviz/phase3a_demo.rviz
```

Terminal 3:

```bash
bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill
bash scripts/final/run_live_gz_screen_recording_demo.sh tray_like_transport
```

## Notes for the Instructor

- Docker headless mode is the reproducible code path.
- Native Gazebo/RViz mode is the visual recording path.
- The provided Google Drive videos show expected visual outputs.
