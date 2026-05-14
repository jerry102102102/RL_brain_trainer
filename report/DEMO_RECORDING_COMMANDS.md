# Demo Recording Commands

Purpose: practical commands for recording the live final demo. The main demo is a runtime demo, not a generated PPT-style video.

## 0. Preflight

Run this before recording:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/check_live_demo_ready.sh
```

This checks ROS2/Gazebo commands, live demo scripts, marker node, RViz config, checkpoints, and L1 fallback artifacts.

## 1. Recommended Live Recording Layout

Use two panes or two windows:

- Left: terminal running `run_live_gz_vlm_demo.sh`.
- Right: Gazebo GUI or RViz.

Optional RViz window:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.bash
rviz2 -d config/rviz/phase3a_demo.rviz
```

RViz should display:

- RobotModel
- TF
- `/v5/demo/target_marker`
- `/v5/cam/overhead/rgb` as optional VLM visual context

The side camera raw feed is hidden by default in the demo RViz config because its fixed simulated side viewpoint is rotated/awkward for presentation. The Gazebo GUI plus RViz target/route markers should be the main recording view.

## 2. Live L1 Bridge Only

Use this if you want to record the natural-language-to-IntentPacket proof first:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/run_live_gz_vlm_demo.sh dry_run_l1
```

Expected output:

- command text
- L1/Qwen bridge status
- IntentPacket
- skill request
- `APPROACH -> FINISHER`

If local Qwen runtime is available:

```bash
USE_QWEN=1 QWEN_BACKEND=qwen_subprocess \
  bash scripts/final/run_live_gz_vlm_demo.sh dry_run_l1
```

If Qwen is unavailable, the default fallback artifact is used and the terminal states that fallback mode is active.

## 3. Demo A: Live Gazebo Local Skill Test

This is the main recording flow. Start Gazebo yourself first so the recording
does not include scene boot noise.

Terminal 1: start the original scene:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
scripts/v5/launch_kitchen_scene.sh --mode gui
```

Terminal 2: run the live L1 -> L2 -> L3 demo:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill
```

This wrapper now defaults to `LAUNCH_SCENE=0`, `HEADLESS=0`, and
`CLEANUP_SCENE=0`. In other words, it does **not** launch or close Gazebo. It
only runs the L1 bridge, target marker publisher, and RL runtime against the
scene you already opened.

By default, this now runs the recording-showcase profile:

```text
--max-targets 5
--target-profile recording_showcase
--approach-steps 60
--finisher-steps 8
```

This uses five visually separated targets so the arm moves lower and across
left/right/front regions of the learned workspace. The live-demo success gates
are intentionally looser than the kinematic report metrics because millimeter
differences are not visible in the final screen recording.

Expected runtime:

- Scene bring-up is handled by Terminal 1.
- Terminal 2 should immediately show L1/MCP, L2/RL, and L3/GZ chain messages.
- The RL rollout prints `[RL]`, `[L2/RL]`, and `[L3/GZ]` progress lines instead of going silent.
- A normal successful run shows lines similar to:

```text
[RL] [phase3a] target 1/4 reachable_fk_target_0 source=default_q_delta_fk:demo_showcase
[RL] [phase3a] target 0: start_pos=0.1097m start_ori=0.1716rad
[RL] [phase3a] target 0: approach_done steps=19 final_pos=0.0017m final_ori=0.0094rad
[RL] [phase3a] target 0: finisher_done steps=8 final_pos=0.0017m final_ori=0.0092rad success=True
```

If you see `[5/6] Starting RL Runtime`, do not stop immediately. Wait for the `[RL]` target/phase lines. The current four-target showcase self-check completed in about 4 minutes on this machine.

Latest verified showcase self-check:

- `report/demo_outputs/live_demo_showcase4_selfcheck_002/final_summary.json`
- `artifacts/v5/phase3a_controlled_sim/live_demo_showcase4_selfcheck_002_controlled_sim/runtime_steps.jsonl`
- success rate: `1.0`
- target count: `4`
- largest tested start position error: approximately `0.332 m`
- mean final position error: approximately `2.01 mm`
- mean final orientation error: approximately `0.0107 rad`
- largest start position error: about `0.219 m`
- largest start orientation error: about `0.406 rad`
- final position error mean: `0.00228 m`
- final orientation error mean: `0.00946 rad`

## 4. Demo B: Mock-Tray Semantic Transport

This is the recommended "tray carrying" demo. It does not claim full
holder1-to-holder8 transport is solved. Instead, L1/Qwen creates a tray-like
semantic waypoint plan that stays inside the verified RL workspace, and L2/L3
executes each waypoint with the learned `Approach -> Finisher` controller.

Terminal 1: start the original scene:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
```

Terminal 2: optionally start RViz so the camera topics and markers are visible:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
rviz2 -d config/rviz/phase3a_demo.rviz
```

Terminal 3: run the mock-tray transport runtime:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/run_live_gz_screen_recording_demo.sh tray_like_transport
```

The terminal should show:

- natural-language task command
- L1/MCP IntentPacket
- generated semantic waypoints:
  `pre_grasp_align`, `under_tray_insert_pose`, `level_lift`, `carry_midline`, `pre_insert_align`, `stable_insert_hold`
- marker publication on `/v5/demo/target_marker`
- L2/RL execution through `Approach -> Finisher`
- L3/GZ trajectory execution logs

The verified numeric fallback for this mock-tray route is:

- waypoint count: `6`
- success rate: `1.0`
- mean final position error: `0.00253 m`
- mean final orientation error: `0.0184 rad`

Artifact:

```text
artifacts/kinematic_phase1/workspace_expansion/tray_like_local_semantic_route_check_20260506/tray_like_local_semantic_route_summary.json
```

For a safer headless validation run:

```bash
RUN_ID=live_demo_local_skill_smoke_001 HEADLESS=1 CLEANUP_SCENE=1 \
  bash scripts/final/run_live_gz_vlm_demo.sh local_skill \
  --max-targets 1 \
  --target-profile smoke \
  --approach-steps 8 \
  --finisher-steps 6 \
  --marker-duration 90
```

Verified output from the existing smoke run:

- `report/demo_outputs/live_demo_local_skill_smoke_001/final_summary.json`
- `report/demo_outputs/live_demo_local_skill_smoke_001/runtime_status.log`
- `artifacts/v5/phase3a_controlled_sim/live_demo_local_skill_smoke_001_controlled_sim/runtime_steps.jsonl`

Latest self-check after script cleanup:

- `report/demo_outputs/live_demo_selfcheck_afterfix_001/final_summary.json`
- `artifacts/v5/phase3a_controlled_sim/live_demo_selfcheck_afterfix_001_controlled_sim/runtime_steps.jsonl`

Verified result:

- success rate: `1.0`
- target count: `1`
- final position error mean: `0.00045 m`
- final orientation error mean: `0.00721 rad`

The scripts use the default ROS2 / FastDDS transport settings.

## 4. Route Prefix Demo

Use this for the route-curriculum segment:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/run_live_gz_vlm_demo.sh route_prefix
```

This mode is allowed to fall back to a kinematic route replay if Gazebo route-prefix execution is unavailable. The fallback is explicit and should be narrated honestly.

## 5. Verified Gazebo Camera Video

Camera-topic recording is optional evidence, not the preferred final recording path. The fixed scene cameras can have poor viewpoints. For the final demo, use screen recording of Gazebo/RViz instead.

An older camera-topic artifact is here:

```text
report/videos/real_gz_camera_phase3a_controlled_sim.mp4
```

To point to the verified artifact without rerunning Gazebo:

```bash
bash scripts/final/run_demo_04_real_gz_sensor_demo.sh
```

To rerun Gazebo and regenerate the camera-topic video:

```bash
RERUN_REAL_GZ_DEMO=1 RUN_ID=final_real_gz_sensor_demo_rerun_001 \
  bash scripts/final/run_demo_04_real_gz_sensor_demo.sh
```

This records `/v5/cam/side/rgb`, runs the frozen `Approach -> Finisher` controlled sim, and writes a runtime summary.

## 5B. Cleanup After Screen Recording

When the recording is finished:

```bash
bash scripts/final/cleanup_live_gz_demo.sh
```

## 6. Health Topics During Recording

Useful checks:

```bash
ros2 topic list | grep -E '/joint_states|/v5/demo/target_marker|/v5/demo/status|/arm_controller'
ros2 topic echo --once /v5/demo/status
ros2 topic echo --once /joint_states
```

The terminal output should show:

- `WAITING_FOR_COMMAND`
- `L1_RESOLVING_INTENT`
- `INTENT_RESOLVED`
- `TARGET_MARKER_PUBLISHED`
- `APPROACH_RUNNING`
- `FINISHER_RUNNING`
- `DONE`

## 7. RViz Camera / VLM Visual Context

To show that the visual side of the stack is connected, open RViz during the demo:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.bash
rviz2 -d config/rviz/phase3a_demo.rviz
```

The RViz config includes:

- `RobotModel`
- `TF`
- `/v5/demo/target_marker`
- `/v5/cam/overhead/rgb`
- `/v5/cam/side/rgb` support, hidden by default because the raw side camera view is not presentation-friendly

If the fixed camera view is not visually ideal, use it as evidence of the vision-topic connection, not as the main demo camera. For the final video, screen-record the Gazebo/RViz viewport that best shows the arm motion.

## 7. What Not To Claim

Do not say full holder1-to-holder8 transport is solved. The correct claim is:

```text
The live demo validates the L1-to-RL-to-Gazebo runtime path, target visualization, and learned local motion. Full scene-level tray transport remains future work.
```
