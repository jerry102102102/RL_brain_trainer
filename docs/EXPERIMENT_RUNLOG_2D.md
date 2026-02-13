# 2D Experiment Run Log (v2 bootstrap)

## Environment
- runner: uv-managed virtual environment (`hrl_ws/.venv`)
- python: 3.12
- torch: 2.10.0+cu128
- cuda: available
- gpu: NVIDIA GeForce RTX 4070

## Commands
```bash
cd hrl_ws
uv sync
PYTHONPATH=src/hrl_trainer uv run python -m hrl_trainer.sim2d.train_sim2d --config src/hrl_trainer/config/exp_2d_a0.yaml --out /tmp/exp_2d_a0_metrics.json
PYTHONPATH=src/hrl_trainer uv run python -m hrl_trainer.sim2d.train_sim2d --config src/hrl_trainer/config/exp_2d_a1.yaml --out /tmp/exp_2d_a1_metrics.json
```

## Metrics (current scaffold)
- A0:
  - success_rate: 0.275
  - time_to_convergence: 28.27
  - tracking_rmse: 3.995
  - recovery_time: 24.2
  - control_effort: 204.60
- A1:
  - success_rate: 0.275
  - time_to_convergence: 28.27
  - tracking_rmse: 3.995
  - recovery_time: 24.2
  - control_effort: 204.60

## Note
A1 is currently a memory-stub config (no real memory module yet), so A0/A1 parity is expected.
Next step: implement tactical memory module and rerun A1.
