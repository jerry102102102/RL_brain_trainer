# Phase 3A Gazebo / ROS2 Reuse Audit

Purpose: identify which existing Gazebo / ROS2 integration pieces should be reused for the first Phase 3A simulation validation path, and which pieces need a thin patch to connect the trained `Approach -> Finisher` kinematic policy line.

## Scope

Phase 3A is not a new RL training phase. It is a migration phase:

```text
Qwen / structured L1 request
-> IntentPacket
-> Approach -> Finisher runtime request
-> ROS2 / Gazebo validation surface
```

The first version is structured-scene / oracle-state oriented. It is not image-grounded yet and does not claim real-robot deployment.

## Directly Reuse

### Scene Bridge

- `scripts/v5/bridge_kitchen_scene.sh`
- Role: creates or validates the symlink bridge from the external kitchen scene repo into `external/kitchen_scene`.
- Current default scene repo: `external/ENPM662_Group4_FinalProject`.
- Phase 3A status: reuse as-is.

### Scene Launch

- `scripts/v5/launch_kitchen_scene.sh`
- Role: launches the bridged kitchen scene in headless or GUI mode and cleans old kitchen-scene processes.
- Phase 3A status: reuse as-is. New Phase 3A demo script calls this script rather than replacing it.

### Canonical Perception / Scene Topic Flow

Current expected flow remains:

```text
/world/empty/pose/info
-> /tray_tracking/pose_stream
-> tray_pose_adapter_node
-> /tray1/pose
-> object_id_publisher_node
-> /v5/perception/object_pose_est
```

Phase 3A keeps this as the expected simulation perception surface. The first runtime path can still use structured scene context, but the ROS health script checks these topics.

### L1 Contract

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/intent_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/qwen_mcp_tools.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/qwen_l1_client.py`

Phase 3A reuses the existing `IntentPacket` validation and Qwen MCP bridge. Qwen still cannot output raw joint control.

### Existing ROS2 Runtime / Executor References

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/task1_train.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/runtime_ros2.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/l3_executor.py`

These modules are useful references for ROS2 joint-state reading, FollowJointTrajectory action handling, and deterministic L3 execution. Phase 3A does not replace them.

## Needs Thin Patch

### Runtime Model Registry

Added:

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/configs/phase3a_runtime_models.yaml`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/runtime_model_registry.py`

Reason: Qwen bridge, runtime node, scripts, and docs must point to one canonical pair of `Approach -> Finisher` checkpoints.

### Phase 3A Runtime Node

Added:

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/phase3a_runtime_node.py`

Role:

- reads a Qwen L1 artifact or direct skill request,
- validates the `IntentPacket`,
- validates the current model registry,
- builds an `APPROACH -> FINISHER` rollout plan,
- optionally publishes high-level JSON messages to `/v5/intent_packet` and `/v5/skill_command` in ROS dry-run mode.

Safety note: this first node does not publish `/arm_controller/joint_trajectory`. L3 remains the owner of executor-level control.

### Phase 3A Scripts

Added:

- `scripts/v5/run_phase3a_demo.sh`
- `scripts/v5/check_phase3a_health.sh`

Role:

- validate scene bridge,
- optionally launch existing kitchen scene,
- run Phase 3A runtime bridge,
- check controller/perception/runtime topic surface.

## Do Not Reuse As Main Path

### Dock-Coarse / Bridge as Mandatory Runtime Modules

Dock-Coarse and Bridge were useful diagnostic experiments, but current Phase 2 evidence supports the simpler:

```text
Approach -> Finisher
```

They should not be reintroduced into Phase 3A unless a Gazebo-specific handoff gap appears.

### Old V5.1 SAC Teacher as Final Runtime Controller

The older SAC teacher / deterministic extraction line remains valuable historical evidence, but Phase 3A should not route through it. The current migration target is the Phase 1/2 kinematic `Approach -> Finisher` result.

### Image-Grounded VLM Perception

Image-grounded Qwen-VL perception is not part of the first Phase 3A validation path. Qwen currently receives structured scene context; image grounding can be added after the ROS/Gazebo runtime path is stable.

## Current Audit Conclusion

The repo already has the main Gazebo bring-up and ROS2 runtime pieces needed for Phase 3A. The missing layer was not another simulator or another RL framework; it was a thin runtime bridge from the Phase 2 `Approach -> Finisher` result into the existing L1/L2/L3 contract surface. That bridge now exists in a safe dry-run / ROS dry-run form.
