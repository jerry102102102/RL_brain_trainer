# Phase 2 Final Demonstration and Results

**Project:** Modular Three-Layer RL for Kitchen Manipulation in ROS 2/Gazebo  
**Report focus:** Phase 2 progress after completing the kinematic arm training stack and adding the first Qwen L1-to-RL bridge  
**Date:** April 2026

## Abstract

Phase 2 advances the project from an operational but incomplete Phase 1 prototype into a working kinematic manipulation research stack with a connected semantic L1 interface. In Phase 1, the custom Gymnasium environment, policy training loop, and deterministic evaluation tools were operational, but the approach-to-docking handoff was unreliable and the VLM component had not started. In Phase 2, the arm-training side was substantially improved: the main controller was simplified to an `Approach -> Finisher` pipeline, and the trained policies now achieve high success across the staged kinematic workspace sweep. In addition, a Qwen-facing MCP bridge was implemented so the L1 model can inspect the scene contract, choose object/source/target slots, resolve a validated `IntentPacket`, and produce an RL-ready high-level skill request.

The project is therefore complete as a kinematic proof-of-concept for the proposed modular architecture. The main remaining work is not conceptual architecture, but migration into a richer Gazebo/ROS visual simulation environment and replacing the current structured scene input with true image-grounded perception.

## 1. Phase 2 Goal

The original project goal was a modular three-layer robot manipulation system:

```text
L1: semantic task interpretation using VLM/LLM
L2: learned local skill policy
L3: deterministic execution and safety
```

For Phase 2, the practical objective was narrowed to a feasible final demonstration:

- Train the kinematic arm to approach the target region.
- Preserve orientation well enough for insertion / docking behavior.
- Keep the policy interface compatible with later Gazebo and VLM integration.
- Connect the L1 model layer to the trained RL stack through a safe structured interface.

This does not yet claim full physical kitchen manipulation. It demonstrates the core modular path: semantic command interpretation can produce a structured intent, and that intent can be converted into a high-level request for the trained arm skill pipeline.

## 2. System Architecture at Phase 2

The current system has two connected parts.

### 2.1 Kinematic RL Skill Stack

The manipulation skill stack is implemented in a pure kinematic Gymnasium environment. The action remains a normalized 7-dimensional joint-delta command, and the observation uses the stable dictionary schema developed in Phase 1.

The final practical policy pipeline is:

```text
Approach policy -> Finisher / Dock policy
```

During Phase 1, a separate Dock-Coarse / Bridge path was explored. Phase 2 ablations showed that once the Approach policy learned to produce a sufficiently clean handoff state, Dock-Coarse was no longer useful as a mandatory middle module and could even degrade a good state. The main path was therefore simplified to `Approach -> Finisher`.

### 2.2 Qwen L1 Bridge

The new L1 bridge exposes a minimal MCP-style tool interface to Qwen:

- `get_l1_scene_context`
- `resolve_intent_packet`
- `prepare_phase1_skill_request`

This lets Qwen act as the semantic interpretation layer without violating the L1/L2/L3 boundary. Qwen is allowed to choose task-level fields such as object, source slot, target slot, and constraints. It is not allowed to output raw joint commands, trajectories, torques, or executor-level commands.

The bridge output is a dry-run high-level skill request:

```text
Qwen output -> IntentPacket -> APPROACH -> FINISHER skill request
```

## 3. Phase 1 Baseline Summary

At the Phase 1 checkpoint, the project was approximately halfway complete:

- The environment, training loop, evaluation harness, and throughput benchmarks were operational.
- Separate approach and dock policies showed measurable progress.
- Switched execution was not reliable.
- The VLM component had not started.

Representative Phase 1 results included:

- Approach policy checkpoints around 71-72% success in the best isolated runs.
- Dock strict success improving from 23% to 44% in representative strict V2 runs.
- Full switched approach+dock success still near 0% in the larger evaluation suite.
- 16-env PPO throughput around 1,762-1,790 aggregate timesteps/s.
- TD3 vectorization scaling from about 272 to 1,658 aggregate timesteps/s from 1 to 8 environments.

These Phase 1 results justified the architecture, but the handoff and VLM pieces were still unresolved.

## 4. Phase 2 RL Results

The most important Phase 2 change was that the handoff problem was reduced and then simplified. Several intermediate directions were tested: Dock-Coarse, Bridge, readiness classification, acceptance mapping, and finisher adaptation. These were useful diagnostic steps, but the final working simplification was:

```text
Approach -> Finisher
```

The latest staged workspace sweep shows that the arm can complete the kinematic target task reliably in the trained range:

| Stage | Success Rate | Handoff Position Error | Handoff Orientation Error | Final Position Error | Final Orientation Error |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.50 mm | 0.0073 rad | 1.67 mm | 0.0106 rad |
| 1 | 1.00 | 0.62 mm | 0.0099 rad | 1.67 mm | 0.0123 rad |
| 2 | 1.00 | 0.85 mm | 0.0119 rad | 1.82 mm | 0.0139 rad |
| 3 | 1.00 | 1.20 mm | 0.0138 rad | 2.14 mm | 0.0164 rad |
| 4 | 1.00 | 1.71 mm | 0.0150 rad | 2.53 mm | 0.0165 rad |
| 5 | 0.93 | 1.96 mm | 0.0177 rad | 2.89 mm | 0.0208 rad |

The Stage 5 result is the most demanding current evaluation stage. It reached 93% success with a mean final position error of about 2.9 mm and mean final orientation error of about 0.021 rad. This is sufficient for the current kinematic proof-of-concept, although more work is needed to validate the same behavior in Gazebo or a real execution stack.

## 5. L1 Qwen-to-RL Demonstration

In Phase 2, the L1 interface was implemented and tested with the local Qwen model setup.

The available local Qwen runtime is:

```text
/home/jerry/.openclaw/workspace/qwenvl
/home/jerry/venvs/qwenvl
Qwen/Qwen2.5-VL-7B-Instruct
```

For the demo command:

```text
Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose.
```

Qwen produced a tool-call JSON object:

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

The MCP bridge resolved this into:

- `IntentPacket` command: `MOVE_PLATE(shelf_A1, shelf_B1)`
- object: `tray1`
- source: `shelf_A1`
- target: `shelf_B1`
- high-level skill pipeline: `APPROACH -> FINISHER`
- target pose: `[-0.92, -1.16, 1.22]`, orientation `[3.14, 0.0, 3.14]`

This demonstrates the first complete semantic-to-RL connection:

```text
Qwen L1 decision -> IntentPacket -> RL-ready skill request
```

The current bridge is intentionally a dry-run interface. It proves the contract and prevents the LLM from directly controlling the robot.

## 6. What Was Completed

By the end of Phase 2, the project completed the following:

- A high-throughput kinematic Gymnasium environment for a 7-DoF arm.
- PPO/TD3 training infrastructure and deterministic evaluation.
- A trained kinematic arm skill pipeline capable of approach and stable pose-preserving insertion behavior.
- A simplified `Approach -> Finisher` final policy path after diagnosing and removing unnecessary intermediate modules.
- A Qwen-facing MCP bridge for L1 semantic command interpretation.
- An end-to-end L1 demo artifact showing Qwen producing an RL-ready skill request.
- Regression tests for the new L1 bridge and existing intent-layer validation.

## 7. Current Limitations

The main limitation is that the final results are still in a pure kinematic environment. This was a deliberate engineering decision because it allowed fast iteration and made the learning problem diagnosable. However, it means the following are not yet solved:

- Full Gazebo physics integration.
- Real camera-based object grounding.
- Contact dynamics, friction, and object interaction.
- ROS controller timing and latency.
- Real robot deployment.

The current Qwen bridge also uses structured scene context. The model can be run as Qwen-VL, but the current final demo does not yet use image-grounded slot/object estimation as the primary source of scene state. The MCP interface is designed so that image-grounded Qwen outputs can be added later without changing the L2/L3 skill contract.

## 8. Interpretation

The most important Phase 2 result is that the project no longer fails at the same point as Phase 1. In Phase 1, the major unsolved issue was approach-to-dock integration. In Phase 2, the kinematic skill pipeline was trained and simplified enough that the arm can reliably complete the target behavior in the trained kinematic workspace. The remaining challenge has shifted from learning the basic arm skill to transferring the system into a richer simulated or real execution environment.

The second important result is that the VLM/LLM layer is no longer only a design idea. Qwen can now produce a structured command that passes through the L1 validation layer and becomes an RL-ready skill request. This directly supports the original modular research claim: semantic understanding and motor skill learning are separated by a strict interface.

## 9. Final Conclusion

Phase 2 demonstrates a working modular prototype. The kinematic arm training is effectively complete for the current simplified task: the policy can approach the target, preserve pose, and complete insertion-style finishing behavior with high success in the trained workspace. The L1 Qwen bridge is also operational at the structured-command level, producing validated `IntentPacket` objects and dry-run RL skill requests.

The project is not yet a complete Gazebo or real-robot kitchen manipulation system. More time is needed to move from kinematic simulation into a richer visual simulation environment. However, the core modular architecture is now demonstrated: a semantic L1 model can issue a safe structured task command, and the trained RL-based L2/L3 stack can interpret that command as a high-level manipulation skill request.

## Evidence Artifacts

- `artifacts/kinematic_phase1/phase1c/workspace_sweep_workspace_noop_vs_previous_summary_001.json`
- `artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json`
- `docs/V5_QWEN_MCP_BRIDGE.md`
- `docs/PHASE1_APPROACH_DOCK_CLOSEOUT.md`
- `hrl_ws/src/hrl_trainer/tests/test_v5_qwen_mcp_bridge.py`
- `hrl_ws/src/hrl_trainer/tests/test_v5_qwen_l1_client.py`

