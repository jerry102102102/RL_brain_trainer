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

## Run

```bash
scripts/v5/run_wp1_5_runtime_parity_check.sh --mode both --timeout-sec 25 --output artifacts/wp1_5/runtime_parity_report.json
```

Notes:

- Use `--bootstrap-scene` when you want the ENPM662 scene packages rebuilt before checks.
- `--mode manual` checks only manual path (scene already running).
- `--mode auto` checks only auto-launch path.

## Report contract

Output is JSON and machine-readable:

- `paths.manual.status`: `PASS|FAIL|BLOCKED`
- `paths.auto.status`: `PASS|FAIL|BLOCKED`
- `parity.status`: `PASS|FAIL|BLOCKED`
- `overall.result`: `PASS|FAIL`

`overall.result=PASS` in `--mode both` means manual and auto-launch paths both produced live samples for all required topics, and per-topic sample results matched across paths.
