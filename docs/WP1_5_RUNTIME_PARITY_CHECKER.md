# WP1.5 Runtime Parity Checker

This checker is part of WP1.5 remediation B and validates simulation startup consistency between:

- manual launch path
- auto-launch path

Required topics:

- `/clock`
- `/joint_states`
- `/v5/cam/overhead/rgb`
- `/v5/cam/side/rgb`
- `/tray1/pose`
- `/v5/perception/object_pose_est`

The checker requires each topic to both appear in `ros2 topic list` and provide at least one live sample within timeout.

Tray source path (WP1.5 patch B):

- Gazebo: `/world/empty/dynamic_pose/info` (`gz.msgs.Pose_V`)
- ROS bridge: `/tray_tracking/pose_stream_raw` (`ros_gz_interfaces/msg/Pose_V`)
- Dedicated extractor output: `/tray1/pose` (`geometry_msgs/msg/PoseStamped`, `frame_id=world`)
- Jazzy limitation (confirmed):
  - `ros_gz_interfaces.msg` missing `Pose_V` Python type in this environment, and
  - `ros_gz_bridge` lacks `Pose_V` template specialization for `/world/empty/dynamic_pose/info`.
- Therefore launch now supports automatic mode switch:
  - `tray_pose_mode=dedicated` when dedicated path is supported.
  - `tray_pose_mode=legacy_degraded` when unsupported and fallback is enabled.

Launch controls:

- `enable_dedicated_tray_source:=true|false` (default `true`)
- `auto_fallback_legacy_on_unsupported:=true|false` (default `true`)
- `enable_legacy_tray_pose_adapter:=true|false` (manual force legacy)

Force modes:

- Force legacy (recommended on Jazzy now):
  - `enable_dedicated_tray_source:=false enable_legacy_tray_pose_adapter:=true`
- Force dedicated (only if environment supports Pose_V end-to-end):
  - `enable_dedicated_tray_source:=true auto_fallback_legacy_on_unsupported:=false`

## Run

```bash
scripts/v5/run_wp1_5_runtime_parity_check.sh --mode both --timeout-sec 25 --output artifacts/wp1_5/runtime_parity_report.json
```

Notes:

- Use `--bootstrap-scene` when you want the ENPM662 scene packages rebuilt before checks.
- `--mode manual` checks only manual path (scene already running).
- `--mode auto` checks only auto-launch path.
- Dedicated tray source validation:
  `ros2 topic echo /tray1/pose --once`
- Default node-path quick check (legacy path disabled):
  `ros2 node list | rg 'tray_pose_extractor|tray_pose_adapter'`
  Expected: `tray_pose_extractor` present, `tray_pose_adapter` absent unless `enable_legacy_tray_pose_adapter:=true`.

## Report contract

Output is JSON and machine-readable:

- `paths.manual.status`: `PASS|FAIL|BLOCKED`
- `paths.auto.status`: `PASS|FAIL|BLOCKED`
- `parity.status`: `PASS|FAIL|BLOCKED`
- `tray_pose_mode`: `dedicated|legacy_degraded|unknown`
- `overall.result`: `PASS|FAIL`

`overall.result=PASS` in `--mode both` means manual and auto-launch paths both produced live samples for all required topics, and per-topic sample results matched across paths.

## M2.2 rollout smoke (RL action schema v2 default)

Run the v5 Rule-L2 tests that exercise rollout schema selection:

```bash
PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m unittest -q hrl_ws/src/hrl_trainer/tests/test_v5_rule_l2_v0.py
```

Expected M2.2 behavior:

- `build_l2_rollout(...)` default path uses action schema `v2` and emits `SkillCommand` with `u_slot_params` + `timing_params`.
- Optional compatibility path remains available with `action_schema='v1'`, which omits v2-only fields.
