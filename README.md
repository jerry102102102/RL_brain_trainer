# RL Brain Trainer

Modular L1-to-RL manipulation research stack for ENPM690.

## Current Truth

This repository is **not** claiming full kitchen manipulation is solved. The current project result is:

```text
Natural language / Qwen L1
-> structured IntentPacket
-> RL skill request
-> kinematic Approach -> Finisher policy stack
-> Gazebo/RViz demo path for visualization and controlled validation
```

The strongest validated motor-control result is still kinematic. Gazebo is used for live demo and integration validation, but full holder1 -> holder8 tray transport with contact dynamics is not solved yet.

## What Works

- **L1 semantic bridge:** Qwen/MCP turns a natural-language command into a validated `IntentPacket` and skill request. L1 does not output raw joint actions or trajectories.
- **Main RL skill path:** the final kinematic skill stack is `Approach -> Finisher`; Dock-Coarse and Bridge were useful diagnostics but are not the main controller.
- **Local kinematic workspace:** Stage 0-4 success is `1.00`; Stage 5 success is `0.93`, with about `2.89 mm` final position error and `0.0208 rad` final orientation error.
- **Workspace expansion:** later curricula extend the home-start shell beyond Stage 5. Stage 10/11 are larger stress shells, not the full reachable workspace.
- **Random-start coverage:** the mixed-start experiment reaches about `80.2%` success in known-workspace random-start eval, while frontier/full-stress regions remain partial.
- **Route curriculum:** the dense holder route improved from longest prefix `21` to stable prefix `120`, with full-route probe longest prefix `170`.
- **Live demo tooling:** scripts exist for L1 -> L2 -> L3 terminal output, RViz target markers, and Gazebo arm motion for recording.

## What Is Not Solved

- Full holder1 -> holder8 tray transport.
- Full continuous workspace coverage.
- Real camera-grounded perception.
- Contact/friction/object dynamics for tray carrying.
- Real robot deployment.

## Main Files To Read

- [Final project summary](report/FINAL_PROJECT_SUMMARY.md)
- [Official artifacts and numbers](report/OFFICIAL_ARTIFACTS.md)
- [Final report PDF](report/FINAL_REPORT.pdf)
- [Final presentation](report/FINAL_PRESENTATION.pptx)
- [Demo recording commands](report/DEMO_RECORDING_COMMANDS.md)
- [Current implementation notes](docs/CURRENT_IMPLEMENTATION.md)
- [Docs index](docs/README.md)

## Quick Commands

Run the L1 semantic bridge with the deterministic mock backend:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source hrl_ws/.venv/bin/activate
export PYTHONPATH=/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer:$PYTHONPATH

python -m hrl_trainer.v5.qwen_l1_client \
  --backend mock_qwen \
  --command "Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose." \
  --output artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request.json
```

Launch the original Gazebo kitchen scene directly:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
```

Run the screen-recording demo after the scene is already open:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill --no-launch-scene
```

Check the latest random-start workspace experiment:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/final/check_full_workspace_randomstart_status.sh workspace_full_coverage_randomstart_overnight_003
```

## Repository Hygiene

Generated training runs, videos, ROS build products, and local demo logs are intentionally ignored. Keep code, configs, runbooks, final report sources, and selected small evidence artifacts in git; keep large checkpoints and runtime outputs local unless explicitly promoted.
