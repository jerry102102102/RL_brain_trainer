# V5 Qwen MCP Bridge

Purpose: define the first safe interface between the Qwen L1 model and the existing V5 / Phase 1 manipulation stack.

This bridge keeps Qwen at the semantic intent layer. Qwen may inspect the scene contract, resolve a task into an `IntentPacket`, and prepare a dry-run Phase 1 skill request. Qwen must not emit joint actions, trajectories, or executor-level commands.

## Layer Boundary

The intended runtime flow is:

```text
Qwen / VLM / LLM
  -> MCP tools
  -> IntentPacket
  -> Approach -> Finisher skill stack
  -> executor / safety
```

The first version is intentionally conservative:

- It exposes the slot map and known objects.
- It resolves structured task proposals into validated `IntentPacket` objects.
- It prepares a high-level `APPROACH -> FINISHER` request in `dry_run` mode.
- It rejects low-level fields such as `joint_trajectory`, `raw_action`, `delta_q`, or `torque`.

## Tools

### `get_l1_scene_context`

Returns:

- known slots
- allowed objects
- available high-level pipeline
- L1 allowed output fields
- forbidden low-level control fields

This is the tool Qwen should call first.

### `resolve_intent_packet`

Input:

```json
{
  "object_id": "tray1",
  "source_slot": "shelf_A1",
  "target_slot": "shelf_B1",
  "constraints": {
    "speed_cap": "SLOW",
    "clearance_m": 0.02,
    "timeout_s": 10.0
  }
}
```

Output:

- validated `IntentPacket`
- canonical `MOVE_PLATE(source, target)` command
- recommendation to call `prepare_phase1_skill_request`

### `prepare_phase1_skill_request`

Input:

```json
{
  "intent_packet": { "...": "..." },
  "dry_run": true
}
```

Output:

- high-level `APPROACH -> FINISHER` request
- target pose selected from the intent packet
- current best Phase 1 policy asset paths
- explicit boundary note

This does not execute the robot yet.

## Running The Server

From the repository root:

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source hrl_ws/.venv/bin/activate
export PYTHONPATH=/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer:$PYTHONPATH

python -m hrl_trainer.v5.qwen_mcp_server
```

For a quick manifest check:

```bash
python -m hrl_trainer.v5.qwen_mcp_server --list-tools
```

For a one-shot local call:

```bash
python -m hrl_trainer.v5.qwen_mcp_server \
  --call-tool resolve_intent_packet \
  --arguments-json '{"object_id":"tray1","source_slot":"shelf_A1","target_slot":"shelf_B1","constraints":{"speed_cap":"SLOW"}}'
```

## End-to-End L1 Demo

The repository also includes a client that runs the full L1-to-RL-input path:

```bash
python -m hrl_trainer.v5.qwen_l1_client \
  --backend qwen_subprocess \
  --command "Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose." \
  --output artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json
```

The repo-tracked Qwen text runner lives at:

```text
hrl_ws/src/hrl_trainer/hrl_trainer/v5/tools/qwenvl_text_runner.py
```

This produces:

1. Qwen model text containing a tool-call JSON object.
2. A parsed `resolve_intent_packet` tool call.
3. A validated `IntentPacket`.
4. A dry-run `APPROACH -> FINISHER` skill request for the trained RL stack.

Known successful demo artifact:

```text
artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json
```

## How Qwen Should Use It

Qwen should follow this sequence:

1. Call `get_l1_scene_context`.
2. Select object/source/target from the scene and user instruction.
3. Call `resolve_intent_packet`.
4. Call `prepare_phase1_skill_request`.
5. Hand the resulting dry-run request to the L2/L3 bridge.

The current bridge is ready for Qwen-as-LLM with structured scene input. Later, the same tool contract can be reused when Qwen receives real images.
