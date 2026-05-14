# Workspace Expansion Audit

Purpose: prepare the next large kinematic training run without touching Gazebo, route curriculum, Dock-Coarse, Bridge, or Qwen.

## Current Official Baseline

- Main skill path: `Approach -> Finisher`
- Official Approach checkpoint: `artifacts/kinematic_phase1/phase1c/approach_finisher_ready_visible_workspace_ft_3m9_001/model_latest.zip`
- Official Finisher checkpoint: `artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip`
- Finisher config: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/dock_workspace_handoff_noop_ft_12env.yaml`
- Stage 0-5 sweep summary: `artifacts/kinematic_phase1/phase1c/workspace_sweep_workspace_noop_vs_previous_summary_001.json`

## Official Stage 0-5 Results

From the Phase 2 final package and workspace sweep artifact:

- Stage 0 success: `1.00`
- Stage 1 success: `1.00`
- Stage 2 success: `1.00`
- Stage 3 success: `1.00`
- Stage 4 success: `1.00`
- Stage 5 success: `0.93`
- Stage 5 final position error: about `0.00289 m`
- Stage 5 final orientation error: about `0.0208 rad`

Interpretation: the local trained kinematic workspace is reliable, but this does not mean the full arm workspace or holder1-to-holder8 route is solved.

## Reusable Code

- Approach PPO training entry: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/training/train_approach_policy.py`
- Kinematic environment: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/envs/arm_kinematic_env.py`
- Curriculum stage sampler: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/envs/curriculum.py`
- Deterministic approach eval: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/eval/eval_approach.py`
- Approach -> Finisher eval: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/eval/eval_approach_finisher.py`
- Existing visible workspace config: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/approach_finisher_ready_visible_workspace_long_12env.yaml`
- Existing larger scene transport config for reference only: `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/approach_scene_transport_large_workspace_xlong_12env.yaml`

## New Pieces Added for This Run

- Workspace stage mix reset support in `envs/reset_samplers.py`
- Workspace expansion gate helpers in `kinematic_phase1/workspace/workspace_curriculum.py`
- Workspace expansion evaluator in `kinematic_phase1/eval/eval_workspace_expansion.py`
- Workspace expansion trainer wrapper in `kinematic_phase1/train_workspace_expansion.py`
- Bigtrain config in `kinematic_phase1/configs/workspace_expansion_bigtrain.yaml`
- Bigtrain launcher in `scripts/final/run_workspace_expansion_bigtrain.sh`
- Status checker in `scripts/final/check_workspace_expansion_status.sh`
- Plotter in `scripts/final/plot_workspace_expansion.py`

## Design Notes

- Stage 0-5 are retained as old workspace replay.
- Stage 6-9 expand the Stage 5 joint-space envelope gradually: mild, medium, large, and stress.
- Training samples a mixture of current stage, previous stages, old Stage 0-5 replay, and a small hard-case slot.
- Checkpoint acceptance is gated by deterministic eval, not by sampled training reward alone.
- `latest` is not treated as final; the intended artifact is `best_checkpoint/model_best_by_gate.zip`.

## Not Used in This Run

- Dock-Coarse / Bridge / readiness classifier
- Route curriculum checkpoints
- Gazebo runtime
- Qwen / MCP bridge
- IK controller

