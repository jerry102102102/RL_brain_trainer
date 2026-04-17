# Task-1 Real L3 Training Loop Rerun Report (2026-03-22)

## Scope delivered
- L3 switched to real constrained execution path (`SafetyConstrainedL3Executor`) instead of stub.
- L2 switched to learnable layer (`LearnableL2Policy`) with replay-driven gain update.
- Replay / update / checkpoint mechanisms integrated into Task-1 training loop.

## Validation
1. `PYTHONPATH=hrl_ws/src/hrl_trainer python3 -m unittest -q hrl_ws/src/hrl_trainer/tests/test_v5_task1_training_bootstrap.py hrl_ws/src/hrl_trainer/tests/test_v5_task1_l3_replay_checkpoint.py`
   - PASS
2. `bash scripts/v5/run_task1_real_l3_training.sh`
   - PASS

## Evidence
- Training artifact: `/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/artifacts/task1_real_l3/task1_training_rows.json`
- Checkpoint: `/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/artifacts/task1_real_l3/task1_checkpoint.json`

## Auto-reset flags and artifact fields
- New CLI flags in `hrl_trainer.v5.task1_train`:
  - `--auto-reset` / `--no-auto-reset` (default for `backend=gazebo`: enabled)
  - `--reset-timeout <seconds>`
  - `--scene-reset-cmd "<shell command>"` (optional pre-arm-reset scene hook)
- Gazebo episode rows now include `reset` metadata with:
  - `applied`, `success`, `duration_ms`, `initial_state` (`q`, `dq`, `ee_proxy`), and `error`
  - For episode `0`, `reset.applied=false` with `skipped_reason=episode_0_bootstrap_runtime_state`
