# Codex Task Pack M1 — 2D Disturbance Experiment Bootstrap

## Goal
Implement and verify the first runnable 2D benchmark scaffold for v2 architecture.

## In-Scope
- Add 2D env with acceleration + steering dynamics
- Add disturbance hooks (sensor noise, delay, friction, impulse, dropout)
- Add heuristic high-level planner interface output
- Add A0/A1 config files
- Add metrics output json

## Out-of-Scope
- Full memory training implementation
- MRAC-TDE-NN full integration
- ROS bridge integration changes

## Definition of Done
1. `docs/EXPERIMENT_FRAMEWORK_2D_V2.md` exists and matches implementation.
2. `hrl_trainer/sim2d/` includes env + planner + eval script.
3. `config/exp_2d_a0.yaml` and `config/exp_2d_a1.yaml` exist.
4. Script prints the required metrics keys:
   - success_rate
   - time_to_convergence
   - tracking_rmse
   - recovery_time
   - control_effort

## Verification Commands
```bash
python3 -m hrl_trainer.sim2d.train_sim2d --config config/exp_2d_a0.yaml
python3 -m hrl_trainer.sim2d.train_sim2d --config config/exp_2d_a1.yaml
```

## Report Back Format
- Files changed
- What was implemented
- Verification output (or blocker if dependency missing)
- Risks / follow-ups
