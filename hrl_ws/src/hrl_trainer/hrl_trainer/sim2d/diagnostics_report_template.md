# RL-brainer Layered Diagnostics Report

Date: `<YYYY-MM-DD>`
Branch: `<branch-name>`
Command: `PYTHONPATH=src/hrl_trainer uv run python -m <module>`

## Summary

| Layer | Check | Status | Key evidence |
|---|---|---|---|
| L0 | RBF straight-line tracking (no learning) | `<PASS/FAIL>` | `speed_rmse=<...>, yaw_rate_rmse=<...>, lateral_rmse=<...>, final_x=<...>, final_y=<...>` |
| L1 | Planner straight-line-consistent subgoals (obstacle-free) | `<PASS/FAIL>` | `invalid_samples=<...>, max_orthogonal_error=<...>, min_progress_ratio=<...>, max_progress_ratio=<...>` |

## Layer Evidence

### L0 Evidence
- Thresholds: `speed_rmse<0.20`, `yaw_rate_rmse<0.10`, `lateral_rmse<0.05`, `abs(final_y)<0.08`, `final_x>1.0`
- Measured: `<paste JSON metrics>`

### L1 Evidence
- Thresholds: `invalid_samples==0`, `max_orthogonal_error<1e-6`, `min_progress_ratio>0`, `max_progress_ratio<=1.01`
- Measured: `<paste JSON metrics>`
