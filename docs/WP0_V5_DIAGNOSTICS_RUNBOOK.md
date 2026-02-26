# WP0 V5 Diagnostics Runbook

This runbook executes the WP0 acceptance checks from `docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md` with reusable JSON outputs for later automation (WP4).

## Config
- Default config: `hrl_ws/src/hrl_trainer/config/v5_wp0_diagnostics.yaml`
- Override any topic/type/threshold via `--config <path>`.

## Environment
- Source ROS 2 + workspace in each terminal.
- Ensure `use_sim_time` is enabled for the publishers/subscribers under test.
- All latency calculations here use: `recv_time - msg.header.stamp` on the same ROS time basis.

## 1. TF checks (`view_frames` + `tf2_echo`)

Command:
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.tf_check_helper
```

Expected output (JSON):
- `ok: true`
- `metrics.checks[0]` is `tf2_tools view_frames`
- `metrics.required_pairs` includes `world -> cam_overhead_optical_frame` and `world -> cam_side_optical_frame`

## 2. State-topic latency gate (P95 < 80 ms, state topics only)

Command (60s window):
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.state_latency_eval --duration-sec 60 --output /tmp/wp0_state_latency.json
```

Pass criteria:
- `metrics.overall.gate.pass == true`
- `metrics.overall.gate.p95_ms_limit == 80.0`
- `metrics.latency_definition == "recv_time - msg.header.stamp (same ROS time basis)"`

## 3. Image health diagnostics (fps/drop/latency)

Command (live publishers):
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.image_health_diag --duration-sec 60 --output /tmp/wp0_image_health_live.json
```

Replay latency command (same tool, replay gate):
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.image_health_diag --duration-sec 60 --replay-mode --output /tmp/wp0_image_health_replay.json
```

Expected fields per camera topic:
- `fps`, `drop.drop_estimate_frames`, `latency.p95_ms`, `latency.gate.pass`
- `camera_info_seen > 0` for required cameras

Notes:
- Replay receive latency gate for image topics uses the same latency definition and checks `P95 < 120 ms` on current environment.

## 4. Approx sync evaluator (overhead + side)

Command:
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.approx_sync_eval --duration-sec 60 --output /tmp/wp0_approx_sync.json
```

Pass criteria:
- `metrics.slop_ms == 50.0`
- `metrics.success_rate > 0.95`
- `metrics.gate.pass == true`

Definition:
- `success_rate = N_pairs / min(N_overhead, N_side)`

## 5. Pose jitter evaluator (static 60s, world frame)

Command:
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.pose_jitter_eval --duration-sec 60 --output /tmp/wp0_pose_jitter.json
```

Pass criteria:
- `metrics.required_frame_id == "world"`
- `metrics.gate.pass == true`
- `metrics.std_xyz_m` each `< 0.003`

Auxiliary metric:
- `metrics.radial_std_m` with definition `std(norm(p - mean(p)))`

## 6. Object id-switch / missing-rate evaluator

Default mode is JSONL (message schema for perception output is repo-specific).

Create sample JSONL (`/tmp/v5_object_id_frames.jsonl`):
```json
{"object_id":"tray1","valid":true}
{"object_id":"tray1","valid":true}
{"object_id":null,"valid":false}
{"object_id":"tray1","valid":true}
```

Command:
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.id_switch_eval --mode jsonl --jsonl /tmp/v5_object_id_frames.jsonl --output /tmp/wp0_id_switch.json
```

Pass criteria:
- `metrics.switch_rate < 0.01`
- `metrics.gate.pass == true`

Warnings:
- `missing_rate >= 0.05` adds a warning (`warn`, not hard fail)

ROS mode (optional once message type is known):
- Set `wp0.id_switch.mode: ros`
- Update `wp0.id_switch.type`, `id_field`, and optional `valid_field`
- Run `python -m hrl_trainer.v5.tools.id_switch_eval --duration-sec 60`

## 7. Rosbag2 record/replay helper

Print helper commands:
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.rosbag2_helper print-commands
```

Record example (terminal A):
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.rosbag2_helper record --bag /tmp/v5_wp0_capture
```

Replay example (after stopping live publishers):
```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m hrl_trainer.v5.tools.rosbag2_helper replay /tmp/v5_wp0_capture
```

## JSON outputs for WP4 reuse
Each tool emits structured JSON with:
- `tool`
- `config_path`
- `ok`
- `metrics`
- `timestamp_unix_ns`

## Local unit tests (pure logic only)
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m unittest discover -s hrl_ws/src/hrl_trainer/tests -p 'test_*.py'
```
