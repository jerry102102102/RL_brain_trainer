# V5 Kitchen Scene Dependency (WP0)

WP0 runtime in this repo depends on the ENPM662 kitchen scene repository and does not vendor scene assets into V5.

Current status (2026-03-09): `artifacts/wp0/wp0_report.json` is overall `PASS` with `6/6` sections passing, including rosbag record/replay latency gate.

## Selected approach
- Strategy: local symlink bridge (`external/kitchen_scene`) + wrapper scripts.
- Reason: minimal/reversible, avoids duplicating large scene assets, keeps source-of-truth in ENPM662 repo.

## Source and bridge paths
- Expected source repo path:
  - `/home/jerry/.openclaw/workspace/repos/personal/ENPM662_Group4_FinalProject`
- Bridge target in this repo:
  - `external/kitchen_scene` (symlink)

## Commands

### 1) Setup/bridge scene
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
scripts/v5/bridge_kitchen_scene.sh \
  --scene-repo /home/jerry/.openclaw/workspace/repos/personal/ENPM662_Group4_FinalProject
```
Expected output:
- success: `OK: bridge created` or `OK: bridge already configured`
- missing source: `ERROR: scene repo not found: ...`

### 2) Launch kitchen scene
```bash
# Dry-run first (prints resolved command)
scripts/v5/launch_kitchen_scene.sh --dry-run

# Actual launch
scripts/v5/launch_kitchen_scene.sh
```
Expected output:
- `Scene repo: .../external/kitchen_scene`
- `Launch cmd: ros2 launch ... use_sim_time:=true`

If auto-detection does not find a launch file, pass explicit command:
```bash
scripts/v5/launch_kitchen_scene.sh \
  --launch-cmd 'ros2 launch <pkg> <launch>.launch.py use_sim_time:=true'
```

### 3) Run WP0 healthcheck
```bash
# Dry-run
scripts/v5/run_wp0_healthcheck.sh --dry-run

# Config-only run (BLOCKED-capable when live data missing)
scripts/v5/run_wp0_healthcheck.sh || true

# Live metrics
scripts/v5/run_wp0_healthcheck.sh --live
```
Expected output:
- prints resolved command line for `hrl_trainer.v5.tools.wp0_healthcheck`
- writes report at `artifacts/wp0/wp0_report.json` unless overridden

## Validation command for missing source repo
```bash
scripts/v5/bridge_kitchen_scene.sh \
  --scene-repo /path/that/does/not/exist \
  --validate-only
```
Expected output:
- `ERROR: scene repo not found: /path/that/does/not/exist`
