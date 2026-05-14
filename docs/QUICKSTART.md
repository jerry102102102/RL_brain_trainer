# Quickstart

This quickstart follows the current final-project path. Older V4/V5.1 commands are preserved in legacy docs, but they are not the main demo path.

## 1. Python Environment

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source hrl_ws/.venv/bin/activate
export PYTHONPATH=/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer:$PYTHONPATH
```

If `.venv` does not exist:

```bash
cd hrl_ws
uv sync
source .venv/bin/activate
```

## 2. L1 Qwen / MCP Semantic Bridge

Mock backend:

```bash
python -m hrl_trainer.v5.qwen_l1_client \
  --backend mock_qwen \
  --command "Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose." \
  --output artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request.json
```

Local Qwen backend, if available:

```bash
python -m hrl_trainer.v5.qwen_l1_client \
  --backend qwen_subprocess \
  --command "Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose." \
  --output artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json
```

Expected contract:

```text
Qwen / L1 -> IntentPacket -> APPROACH -> FINISHER skill request
```

L1 must not output raw joint commands.

## 3. Gazebo / RViz Recording Path

Terminal 1: launch the original external scene directly.

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
```

Terminal 2: launch RViz.

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
rviz2 -d config/rviz/phase3a_demo.rviz
```

Terminal 3: run the demo without relaunching the scene.

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill --no-launch-scene
```

Planning showcase mode:

```bash
bash scripts/final/run_live_gz_screen_recording_demo.sh tray_like_transport --no-launch-scene
```

## 4. Final Package Checks

```bash
bash scripts/final/check_final_package.sh
bash scripts/final/check_live_demo_ready.sh
```

Workspace random-start status:

```bash
bash scripts/final/check_full_workspace_randomstart_status.sh workspace_full_coverage_randomstart_overnight_003
```

## 5. Current Claims

This repo demonstrates:

- Qwen semantic L1 bridge,
- kinematic `Approach -> Finisher` RL skill stack,
- route-curriculum prefix improvement,
- workspace expansion and random-start coverage experiments,
- Gazebo/RViz live demo path.

This repo does not claim:

- full holder1 -> holder8 transport solved,
- full continuous workspace solved,
- real robot deployment complete.
