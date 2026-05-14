# Current Implementation Truth

Last consolidated for main-branch cleanup.

## System Story

The project is a modular L1-to-RL manipulation prototype:

```text
Natural language command
-> Qwen / MCP semantic bridge
-> IntentPacket
-> RL skill request
-> kinematic Approach -> Finisher
-> Gazebo/RViz demo visualization and controlled execution path
```

The final motor-control mainline is **Approach -> Finisher**. Dock-Coarse, Bridge, acceptance maps, and readiness classifiers were diagnostic experiments that helped remove unnecessary modules from the final path.

## L1 / L2 / L3 Contract

- **L1:** Qwen/MCP resolves semantic intent, object ID, source/target slots, constraints, and skill pipeline.
- **L2:** learned policy stack performs rollout and skill-level decisions.
- **L3:** deterministic execution/safety layer sends bounded control to ROS2/Gazebo.

Hard boundary:

```text
L1 must not output raw joint actions, torques, or joint trajectories.
```

## Current Kinematic Skill Result

Official local workspace result:

| Stage | Success | Final Pos Error | Final Ori Error |
|---:|---:|---:|---:|
| 0 | 1.00 | 1.67 mm | 0.0106 rad |
| 1 | 1.00 | 1.67 mm | 0.0123 rad |
| 2 | 1.00 | 1.82 mm | 0.0139 rad |
| 3 | 1.00 | 2.14 mm | 0.0164 rad |
| 4 | 1.00 | 2.53 mm | 0.0165 rad |
| 5 | 0.93 | 2.89 mm | 0.0208 rad |

Interpretation: the local kinematic skill works reliably inside the trained workspace. This does not imply full workspace or full tray transport.

## Workspace Expansion

Later training pushed beyond Stage 5. The best home-start expansion shell reaches useful behavior through Stage 8 and partial behavior in Stage 9-11, but Stage 10/11 are still **manual stress shells**, not the full continuous reachable workspace.

Random-start / mixed-start coverage experiment:

| Eval Split | Success | Mean Pos Error | Mean Ori Error | Main Failure |
|---|---:|---:|---:|---|
| Known workspace random-start | 0.802 | 3.87 mm | 0.0185 rad | position / timeout |
| Frontier random-start | 0.240 | 260 mm | 0.611 rad | position |
| Full reachable stress | 0.219 | 424 mm | 1.156 rad | position |

Interpretation: the policy is beginning to behave like a mixed-start goal-conditioned controller in known regions, but frontier/full-reachable coverage is still partial.

## Route Curriculum

The route curriculum uses dense `q_goal` targets as policy observations, not as direct controller commands.

Official result:

| Metric | Value |
|---|---:|
| Baseline full483 success fraction | 0.0435 |
| Baseline longest prefix | 21 |
| Prefix120 sequential success | 1.0 |
| Prefix120 longest prefix | 120 |
| Full483 probe success fraction | 0.4741 |
| Full483 probe longest prefix | 170 |
| First failure index | 171 |
| First failure reason | position |

`full483 success = 0.4741` is a waypoint-level/probe fraction, not a 47.41% probability of completing the whole holder1 -> holder8 route.

## Gazebo / RViz Demo Status

The Gazebo demo path is for visual proof of integration:

- natural language command is resolved into an L1 skill request,
- target markers and route/subtask markers are published,
- terminal shows L1 -> L2 -> L3 state transitions,
- the arm moves in Gazebo using the demo runtime path.

Current demo claims:

- L1 semantic contract works.
- RViz/Gazebo visualization works for target/subtask display.
- Local learned-policy motion can be shown.

Current demo limitations:

- full holder1 -> holder8 tray transport is not solved,
- object contact/friction is not validated,
- camera view is for demonstration/context, not robust vision grounding,
- Gazebo launch should use the original external scene launch directly when recording.

## Recommended Recording Flow

Terminal 1:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
```

Terminal 2:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
rviz2 -d config/rviz/phase3a_demo.rviz
```

Terminal 3:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill --no-launch-scene
```

Alternative planning showcase:

```bash
bash scripts/final/run_live_gz_screen_recording_demo.sh tray_like_transport --no-launch-scene
```
