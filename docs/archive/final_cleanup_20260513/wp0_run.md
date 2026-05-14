# WP0 Run (Copy/Paste)

## Setup
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
export PYTHONPATH=hrl_ws/src/hrl_trainer
CONFIG=hrl_ws/src/hrl_trainer/config/wp0_config.yaml
ART=artifacts/wp0
mkdir -p "$ART"
```

## 1) Bridge ENPM662 kitchen scene dependency (required)
```bash
scripts/v5/bridge_kitchen_scene.sh \
  --scene-repo /home/jerry/.openclaw/workspace/repos/personal/ENPM662_Group4_FinalProject
```
Expected output:
- `OK: bridge created` (or `OK: bridge already configured`)

## 2) Launch kitchen scene (required before live WP0)
```bash
# Auto-detect kitchen launch file, or pass --launch-cmd explicitly if needed.
scripts/v5/launch_kitchen_scene.sh --dry-run
scripts/v5/launch_kitchen_scene.sh
```
Expected output:
- `Scene repo: .../external/kitchen_scene`
- `Launch cmd: ros2 launch ... use_sim_time:=true`

## 3) Unified healthcheck (dry run, produces valid BLOCKED-capable report)
```bash
scripts/v5/run_wp0_healthcheck.sh --dry-run
scripts/v5/run_wp0_healthcheck.sh || true
```

## 4) Unified healthcheck (live metrics)
```bash
scripts/v5/run_wp0_healthcheck.sh --live
```

## 5) Manual rosbag record
```bash
python3 -m hrl_trainer.v5.tools.rosbag2_helper --config "$CONFIG" record --bag /tmp/v5_wp0_capture
```

## 6) Replay bag + replay latency check in one command
```bash
scripts/v5/run_wp0_healthcheck.sh --live --replay-bag /tmp/v5_wp0_capture
```

## 7) Optional: print rosbag commands only
```bash
python3 -m hrl_trainer.v5.tools.rosbag2_helper --config "$CONFIG" print-commands
```

## 8) Inspect result quickly
```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/wp0/wp0_report.json")
r = json.loads(p.read_text())
print("overall:", r["overall"]["result"])
for k,v in r["sections"].items():
    print(f"{k}: {v['status']}")
PY
```
