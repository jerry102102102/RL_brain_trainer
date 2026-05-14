# HRL Workspace

This workspace contains the Python package used by the project. The active final-project line is the kinematic `Approach -> Finisher` stack plus Qwen/L1 and Phase 3A demo integration.

## Active Paths

- `hrl_trainer.kinematic_phase1`: Gymnasium-style kinematic arm environment, rewards, PPO/TD3 training, deterministic eval, workspace expansion, route curriculum, and random-start coverage tools.
- `hrl_trainer.v5`: Qwen/MCP L1 bridge, Phase 3A runtime/demo helpers, target markers, tray-like waypoint planning, and Gazebo controlled-sim utilities.
- `hrl_trainer.v5_1`: older SAC/deterministic-extraction research path retained for reference, not the final ENPM690 demo mainline.
- `hrl_trainer.sim2d`: legacy 2D baseline retained for historical comparison only.

## Setup

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source hrl_ws/.venv/bin/activate
export PYTHONPATH=/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer:$PYTHONPATH
```

If the virtual environment is missing, recreate it from the project root:

```bash
cd hrl_ws
uv sync
```

## L1 Qwen Bridge

Mock backend:

```bash
python -m hrl_trainer.v5.qwen_l1_client \
  --backend mock_qwen \
  --command "Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose." \
  --output artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request.json
```

Local Qwen backend, when available:

```bash
python -m hrl_trainer.v5.qwen_l1_client \
  --backend qwen_subprocess \
  --command "Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose." \
  --output artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json
```

L1 produces semantic intent and structured skill requests only. It must not produce raw joint actions, joint trajectories, torques, or `delta_q`.

## Kinematic Training And Evaluation

Important current modules:

- Workspace expansion trainer: `hrl_trainer.kinematic_phase1.train_workspace_expansion`
- Route curriculum trainer: `hrl_trainer.kinematic_phase1.train_route_curriculum`
- Workspace expansion eval: `hrl_trainer.kinematic_phase1.eval.eval_workspace_expansion`
- Full workspace random-start eval: `hrl_trainer.kinematic_phase1.eval.eval_full_workspace_coverage`
- Route eval: `hrl_trainer.kinematic_phase1.eval.eval_route_curriculum`

Status helpers live in `../scripts/final/`.

## Gazebo Demo Integration

For recording, launch the original external scene first, then run the demo script from the repo root:

```bash
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
```

```bash
bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill --no-launch-scene
```

The demo path is meant to show the L1 -> L2 -> L3 contract, target visualization, and learned-policy motion. It is not a claim that full tray transport is solved.

## Documentation

Start from:

- `../README.md`
- `../docs/README.md`
- `../docs/CURRENT_IMPLEMENTATION.md`
- `../report/OFFICIAL_ARTIFACTS.md`
