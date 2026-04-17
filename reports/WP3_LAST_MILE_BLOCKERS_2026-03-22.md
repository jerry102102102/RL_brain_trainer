# WP3 Last Mile Blockers (2026-03-22)

## Result summary
- Tried non-mock evidence generation with `--mode real`.
- Evidence was produced under `artifacts/wp3/hil_dryrun/2026-03-22_161601/` and includes `"mode": "real"`.
- WS1 still failed in `real` evidence mode because runtime checks could not be satisfied in this environment.

## Precise blocker (single root blocker)
**No reachable real HIL runtime stack on this host (ROS 2 runtime + bridge process + robot/sim topics).**

Observed concrete failures from generated evidence/log:
- `ros2 CLI not found in PATH`
- `no parameter_bridge/l3_runtime_bridge process found`
- therefore checks `health/topic/bridge` are all false and `pass=false`

## Missing dependencies / hardware / services (concrete)
1. **ROS 2 CLI/runtime missing**
   - Required command: `ros2`
   - Verify: `command -v ros2`
2. **Runtime bridge process not running**
   - Required process pattern: `parameter_bridge` or `l3_runtime_bridge`
   - Verify: `pgrep -fa 'parameter_bridge|l3_runtime_bridge'`
3. **Live runtime topics/nodes missing**
   - Required probe commands:
     - `ros2 node list`
     - `ros2 topic list`
   - Expected at least one runtime topic such as `/tf` or `/clock`

## 30-60 minute minimal handoff steps for Jerry
1. Start/attach to the real HIL stack in the same shell (source ROS env, launch bridge/runtime, connect robot/sim).
2. Verify runtime is alive:
   - `command -v ros2`
   - `ros2 node list`
   - `ros2 topic list | egrep '^/tf$|^/clock$'`
   - `pgrep -fa 'parameter_bridge|l3_runtime_bridge'`
3. Re-generate real evidence and rerun WP3 gates in real mode:
   - `scripts/wp3_run_gates.sh --with-hil-dryrun --hil-mode real --hil-policy rule_l2_v0 --hil-seed 42 --hil-evidence-mode real`

If step 3 exits 0, WP3 real-runtime-evidence gate is complete.
