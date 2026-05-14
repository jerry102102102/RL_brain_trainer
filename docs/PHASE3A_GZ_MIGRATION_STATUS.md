# Phase 3A Gazebo Migration Status

Purpose: summarize the first migration step from the pure kinematic `Approach -> Finisher` result into the existing Gazebo / ROS2 / Qwen L1-L2-L3 integration path.

## Current Goal

Phase 3A is the first simulation validation path:

```text
Qwen L1 request
-> validated IntentPacket
-> Approach -> Finisher runtime plan
-> ROS2 / Gazebo contract topics
```

This phase does not retrain the policy and does not introduce a new kinematic framework. It reuses the current Gazebo bring-up and the trained `Approach -> Finisher` assets.

## Reused Gazebo / ROS2 Components

Directly reused:

- `scripts/v5/bridge_kitchen_scene.sh`
- `scripts/v5/launch_kitchen_scene.sh`
- existing canonical topic flow:
  - `/world/empty/pose/info`
  - `/tray_tracking/pose_stream`
  - `/tray1/pose`
  - `/v5/perception/object_pose_est`
- L1/L2/L3 topic contract:
  - `/v5/intent_packet`
  - `/v5/skill_command`
  - `/arm_controller/joint_trajectory`

The new Phase 3A scripts build on the existing scripts instead of replacing them.
The kitchen scene source is expected at `external/ENPM662_Group4_FinalProject`,
with `scripts/v5/bridge_kitchen_scene.sh` validating or creating the
`external/kitchen_scene` symlink.

## Current Runtime Model Registry

Single source of truth:

```text
hrl_ws/src/hrl_trainer/hrl_trainer/v5/configs/phase3a_runtime_models.yaml
```

Current Phase 3A mainline:

- Approach checkpoint:
  `artifacts/kinematic_phase1/phase1c/approach_finisher_ready_visible_workspace_ft_3m9_001/model_latest.zip`
- Finisher checkpoint:
  `artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip`

The current Approach was fine-tuned from the Phase 1/2 closeout local
finisher-ready checkpoint to improve the larger visible-workspace Gazebo stress
test.  The previous local checkpoint remains useful as the small-range baseline:
`artifacts/kinematic_phase1/phase1c/approach_finisher_ready_v2_settle_ft_786k_001/model_latest.zip`.

## New Runtime Node

Added:

```text
hrl_ws/src/hrl_trainer/hrl_trainer/v5/phase3a_runtime_node.py
```

The node currently supports:

- reading a full Qwen L1 artifact or direct skill request JSON,
- validating the embedded `IntentPacket`,
- validating the model registry paths,
- constructing an `APPROACH -> FINISHER` state-machine rollout plan,
- optional `ros_dry_run` publishing to `/v5/intent_packet` and `/v5/skill_command`.

Safety boundary:

- L1 does not output raw control.
- Phase 3A runtime does not publish `/arm_controller/joint_trajectory` in this first path.
- L3 remains responsible for executor-level trajectory output.

## Controlled Sim Runtime

Added:

```text
hrl_ws/src/hrl_trainer/hrl_trainer/v5/phase3a_controlled_sim.py
scripts/v5/run_phase3a_controlled_sim.sh
```

This is the first controlled Gazebo validation path for the frozen
`Approach -> Finisher` policies.  It:

- reads current arm state from `/joint_states`,
- builds reachable FK target poses from small joint-space target offsets,
- publishes a visible target marker into Gazebo through the existing
  `GazeboTargetVisualizer`,
- rolls out frozen Approach and Finisher policies,
- sends each joint target through the existing `RuntimeROS2Adapter` / L3
  trajectory path,
- writes per-step JSONL logs with pose error, action magnitude, commanded
  joint delta, actual joint delta, tracking error, execution status, handoff
  status, and strict-pose status.

This path is intentionally not an L1 raw-control path.  L1/Qwen still produces
semantic intent and high-level requests only.  Policy inference and trajectory
publication remain in runtime / L2-L3 controlled code.

## Qwen Dry-Run Skill Request

The existing Qwen bridge now feeds Phase 3A:

```text
artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json
```

The known demo command is:

```text
Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose.
```

Qwen produced:

```json
{
  "tool": "resolve_intent_packet",
  "arguments": {
    "object_id": "tray1",
    "source_slot": "shelf_A1",
    "target_slot": "shelf_B1",
    "constraints": {
      "speed_cap": "SLOW"
    }
  }
}
```

The runtime bridge converts this into a validated `APPROACH -> FINISHER` Phase 3A plan.

## Demo Commands

Dry-run runtime plan:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source hrl_ws/.venv/bin/activate
export PYTHONPATH=/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer:$PYTHONPATH

python -m hrl_trainer.v5.phase3a_runtime_node \
  --request-json artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json \
  --output-json artifacts/v5/phase3a_demo/phase3a_runtime_plan.json \
  --mode dry_run
```

Scripted path:

```bash
scripts/v5/run_phase3a_demo.sh --skip-bridge-validation
```

The `--skip-bridge-validation` option is intended for runtime-only checks on
machines where the external kitchen scene repository is not checked out yet.
Without that flag, the script validates the existing scene bridge first. The
script-level `--dry-run` flag only prints commands and does not write a runtime
plan.

ROS dry-run after scene bring-up:

```bash
scripts/v5/run_phase3a_demo.sh --launch-scene --mode ros_dry_run
```

When `--launch-scene` is used, the demo script first ensures the
`external/kitchen_scene -> external/ENPM662_Group4_FinalProject` bridge exists,
then starts `scripts/v5/launch_kitchen_scene.sh` with `nohup` so the scene
remains alive after the runtime dry-run command exits. The script sources
`/opt/ros/jazzy/setup.bash` and the scene install setup automatically when
available.

Health check:

```bash
scripts/v5/check_phase3a_health.sh
```

Controlled Gazebo policy rollout:

```bash
scripts/v5/run_phase3a_controlled_sim.sh \
  --launch-scene \
  --scene-mode gui \
  --max-targets 2 \
  --run-id controlled_sim_gui_demo_001
```

Visible 20-target stress rollout:

```bash
scripts/v5/run_phase3a_controlled_sim.sh \
  --launch-scene \
  --scene-mode gui \
  --max-targets 20 \
  --target-profile visible_workspace \
  --approach-steps 24 \
  --finisher-steps 12 \
  --run-id controlled_sim_gui_visible20_001
```

If the current WSL shell does not expose a GUI display (`DISPLAY` /
`WAYLAND_DISPLAY` empty), use the same runtime path in headless mode:

```bash
scripts/v5/run_phase3a_controlled_sim.sh \
  --launch-scene \
  --scene-mode headless \
  --cleanup-scene \
  --max-targets 2 \
  --run-id controlled_sim_headless_two_targets_001
```

## Current Status

Completed:

- Reuse audit completed.
- Phase 3A model registry added.
- Qwen MCP defaults now read from the registry.
- Phase 3A runtime node added.
- Phase 3A demo / health scripts added.
- Unit tests added for the runtime bridge.

Validated on May 3, 2026:

- `python -m unittest discover -s hrl_ws/src/hrl_trainer/tests -p 'test_v5*.py'`
  passed: 157 tests.
- `scripts/v5/bridge_kitchen_scene.sh --validate-only` passed against the
  vendored scene at `external/ENPM662_Group4_FinalProject`.
- `scripts/v5/launch_kitchen_scene.sh --dry-run --mode headless` produced the
  expected Gazebo launch command.
- `python -m hrl_trainer.v5.phase3a_runtime_node ... --mode dry_run` produced
  an `APPROACH_FINISHER` plan with both checkpoint paths present.
- `python -m hrl_trainer.v5.phase3a_runtime_node ... --load-policies` loaded
  both Approach and Finisher checkpoints as PPO policies on CPU.
- `scripts/v5/run_phase3a_demo.sh --launch-scene --mode ros_dry_run` completed
  with `ros_dry_run_published=true` while preserving the safety boundary:
  runtime does not publish `/arm_controller/joint_trajectory`.
- `scripts/v5/check_phase3a_health.sh` passed while the scene was running:
  `arm_controller` and `joint_state_broadcaster` were active, `/joint_states`,
  `/tray1/pose`, and `/v5/perception/object_pose_est` produced samples.
- `scripts/v5/run_phase3a_controlled_sim.sh --launch-scene --scene-mode headless
  --max-targets 2 --run-id controlled_sim_headless_two_targets_001` completed
  a real Gazebo controller rollout with frozen RL policies:
  - target count: 2
  - success rate: 1.0
  - handoff confirmed rate: 1.0
  - mean final position error: 0.00148 m
  - mean final orientation error: 0.0153 rad
  - step log:
    `artifacts/v5/phase3a_controlled_sim/controlled_sim_headless_two_targets_001/runtime_steps.jsonl`
  - summary:
    `artifacts/v5/phase3a_controlled_sim/controlled_sim_headless_two_targets_001/controlled_sim_summary.json`
- `scripts/v5/run_phase3a_controlled_sim.sh --launch-scene --scene-mode headless
  --max-targets 20 --target-profile visible_workspace --approach-steps 24
  --finisher-steps 12 --run-id controlled_sim_headless_visible20_001`
  completed a larger visible-motion stress rollout:
  - target count: 20
  - start position error mean / max: 0.217 m / 0.352 m
  - final position error mean / max: 0.0450 m / 0.140 m
  - success rate under strict 5 mm / 0.05 rad success: 0.30
  - handoff confirmed rate: 0.05
  - interpretation: the controller clearly moves away from the start and
    reduces large target error, but the frozen policy is not yet robust across
    this wider visible workspace without additional training or a larger
    rollout budget.
- After visible-workspace kinematic fine-tuning:
  `scripts/v5/run_phase3a_controlled_sim.sh --launch-scene --scene-mode headless
  --max-targets 20 --target-profile visible_workspace --approach-steps 72
  --finisher-steps 12 --run-id controlled_sim_headless_visible20_newapproach_ap72_001`
  completed with:
  - target count: 20
  - success rate: 0.70
  - handoff confirmed rate: 0.45
  - mean final position error: 0.00440 m
  - mean final orientation error: 0.0344 rad
  - training summary:
    `artifacts/kinematic_phase1/phase1c/approach_finisher_ready_visible_workspace_ft_3m9_001/training_summary.json`
  - Gazebo summary:
    `artifacts/v5/phase3a_controlled_sim/controlled_sim_headless_visible20_newapproach_ap72_001/controlled_sim_summary.json`

Not yet complete:

- GUI validation from the Codex shell was not possible because this shell had
  no GUI display variables (`DISPLAY` / `WAYLAND_DISPLAY` were empty).  The GUI
  command is documented above and should be run from a WSLg-capable terminal.
- L2 JSON `SkillCommand` adapter into a production long-running subscriber.
- Broader workspace target schedule and controller timing stress tests.
- Contact / insertion dynamics.
- Image-grounded Qwen-VL perception.
- Real robot validation.

## Phase 3A Interpretation

The current Phase 3A result is now past pure dry-run: it includes a controlled
Gazebo rollout where the frozen `Approach -> Finisher` policies read live
`/joint_states`, publish target markers, and execute through the existing
controller path.  The result supports the interpretation that the kinematic
policy behavior transfers to small structured Gazebo FK targets, while broader
workspace, contact, and image-grounded perception remain future validation work.
