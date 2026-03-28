# V5.1 Single-Path Delivery

Date: 2026-03-27

## Scope
Converge V5.1 to a single runtime path:
- Policy: SAC Torch only (`policy_mode=sac_torch`)
- Environment: ROS2 Jazzy + `hrl_ws/.venv/bin/python`

## Delivered

### A) Code convergence
- `pipeline_e2e` now rejects any mode except `sac_torch`.
- Removed legacy numpy SAC wiring from `pipeline_e2e`.
- Removed `v5_1/sac_agent.py` from mainline.
- Updated `v5_1/__init__.py` exports to avoid numpy SAC symbols.

### B) Environment lock
Added:
- `scripts/v5_1/activate_env.sh`
  - `source /opt/ros/jazzy/setup.zsh`
  - sources external install setup if present
  - activates `hrl_ws/.venv`
  - exports `PYTHONPATH=hrl_ws/src/hrl_trainer`
- `scripts/v5_1/env_check.sh`
  - verifies python executable is `hrl_ws/.venv/bin/python`
  - checks `torch` and `rclpy`
  - fails fast on mismatch

### C) Tests
- Removed legacy numpy SAC test (`test_v5_1_sac_agent.py`).
- Updated pipeline e2e tests for torch-only mode.
- Kept torch/reward/gates route tests.

### D) Docs
- README updated with single-path guidance and env scripts.
- V5.1 docs updated to reflect torch-only line.

## Validation commands
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source scripts/v5_1/activate_env.sh
scripts/v5_1/env_check.sh

# test suite subset (torch path)
hrl_ws/.venv/bin/python -m pytest -q \
  hrl_ws/src/hrl_trainer/tests/test_v5_1_pipeline_e2e.py \
  hrl_ws/src/hrl_trainer/tests/test_v5_1_sac_torch.py \
  hrl_ws/src/hrl_trainer/tests/test_v5_1_reward.py \
  hrl_ws/src/hrl_trainer/tests/test_v5_1_gates.py
```

## Risks / follow-up
- `sac_torch` dependency is now hard requirement; environments without torch cannot run V5.1 pipeline.
- Some historical planning docs still mention `rule|sac`; these are archival/planning references and should be cleaned in a follow-up doc sweep.
