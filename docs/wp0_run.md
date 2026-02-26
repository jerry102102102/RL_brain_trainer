# WP0 Run (Copy/Paste)

## Setup
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
export PYTHONPATH=hrl_ws/src/hrl_trainer
CONFIG=hrl_ws/src/hrl_trainer/config/wp0_config.yaml
ART=artifacts/wp0
mkdir -p "$ART"
```

## 1) Launch sim / stack (example placeholder)
```bash
# Replace with your actual sim + publishers launch command for this branch/workspace.
# Requirement: topics in $CONFIG must be live and use sim time.
ros2 launch <your_pkg> <your_launch>.launch.py use_sim_time:=true
```

## 2) Unified healthcheck (dry run, produces valid BLOCKED-capable report)
```bash
python3 -m hrl_trainer.v5.tools.wp0_healthcheck --config "$CONFIG" --artifacts-dir "$ART" --output "$ART/wp0_report.json" || true
```

## 3) Unified healthcheck (live metrics)
```bash
python3 -m hrl_trainer.v5.tools.wp0_healthcheck --live --config "$CONFIG" --artifacts-dir "$ART" --output "$ART/wp0_report.json"
```

## 4) Manual rosbag record
```bash
python3 -m hrl_trainer.v5.tools.rosbag2_helper --config "$CONFIG" record --bag /tmp/v5_wp0_capture
```

## 5) Replay bag + replay latency check in one command
```bash
python3 -m hrl_trainer.v5.tools.wp0_healthcheck --live --replay-bag /tmp/v5_wp0_capture --config "$CONFIG" --artifacts-dir "$ART" --output "$ART/wp0_report.json"
```

## 6) Optional: print rosbag commands only
```bash
python3 -m hrl_trainer.v5.tools.rosbag2_helper --config "$CONFIG" print-commands
```

## 7) Inspect result quickly
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

