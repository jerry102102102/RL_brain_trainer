# WP1.5 Runtime Parity Checker

This checker is part of WP1.5 remediation B and validates simulation startup consistency between:

- manual launch path
- auto-launch path

Required topics:

- `/clock`
- `/joint_states`
- `/v5/cam/overhead/rgb`
- `/v5/cam/side/rgb`

The checker requires each topic to both appear in `ros2 topic list` and provide at least one live sample within timeout.

Tray source path (WP1.5 patch B):

- Gazebo: `/world/empty/dynamic_pose/info` (`gz.msgs.Pose_V`)
- ROS bridge: `/tray_tracking/pose_stream_raw` (`ros_gz_interfaces/msg/Pose_V`)
- Dedicated extractor output: `/tray1/pose` (`geometry_msgs/msg/PoseStamped`, `frame_id=world`)
- Legacy fallback path (`/tray_tracking/pose_stream` -> `tray_pose_adapter_node`) is disabled by default and can be enabled with `enable_legacy_tray_pose_adapter:=true`.

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
- `overall.result`: `PASS|FAIL`

`overall.result=PASS` in `--mode both` means manual and auto-launch paths both produced live samples for all required topics, and per-topic sample results matched across paths.
