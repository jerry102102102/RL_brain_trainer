# Repo Cleanup Notes For Main

Purpose: document what should be kept, ignored, archived, or deleted before pushing this project to `main`.

## Keep In Git

- Core source code under `hrl_ws/src/hrl_trainer/`.
- Current configs under `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/` and `hrl_ws/src/hrl_trainer/hrl_trainer/v5/configs/`.
- Demo/run scripts under `scripts/final/` and `scripts/v5/`.
- Final package docs under `report/`.
- Current docs:
  - `docs/README.md`
  - `docs/CURRENT_IMPLEMENTATION.md`
  - `docs/QUICKSTART.md`
  - `docs/V5_QWEN_MCP_BRIDGE.md`
  - `docs/PHASE3A_GZ_REUSE_AUDIT.md`
  - `docs/PHASE3A_GZ_MIGRATION_STATUS.md`
  - `docs/RL_WORKSPACE_AND_TRANSPORT_STATUS.md`
  - `docs/ROUTE_CURRICULUM_TRAINING_PLAN.md`

## Keep Local / Do Not Commit

- `build/`, `install/`, `log/`
- `hrl_ws/.venv/`
- `external/ENPM662_Group4_FinalProject/`
- `external/kitchen_scene`
- `artifacts/kinematic_phase1/` large training runs and checkpoints
- `report/demo_outputs/`
- `report/videos/`
- `report/video_frames/`
- downloaded duplicate `reports/` folder
- `*:Zone.Identifier`

These are now covered by `.gitignore` where appropriate.

## Legacy But Useful

Older docs remain for traceability but are not current truth:

- V4 / V5.1 / WP0 / WP1.5 design documents
- Dock-Coarse and Bridge planning notes
- older Phase 1/Phase 1C diagnostics
- literature notes

Use `docs/README.md` and `docs/CURRENT_IMPLEMENTATION.md` as the current entry point.

## Current Main-Branch Claim

The repository should claim:

- modular L1-to-RL architecture demonstrated,
- Qwen L1 bridge produces safe structured skill requests,
- kinematic `Approach -> Finisher` works in the trained workspace,
- workspace expansion and random-start coverage are progressing,
- route curriculum improves route-prefix following,
- Gazebo/RViz demo path can visualize and run controlled local motion.

The repository should not claim:

- full kitchen manipulation solved,
- full holder1 -> holder8 transport solved,
- full continuous workspace solved,
- real robot deployment complete.

## Recommended Pre-Push Checks

```bash
git status --short
bash scripts/final/check_final_package.sh
bash scripts/final/check_live_demo_ready.sh
bash scripts/final/check_full_workspace_randomstart_status.sh workspace_full_coverage_randomstart_overnight_003
```

If Gazebo demo recording is needed, start the original external scene manually first; do not rely on wrapper scene launch for the final recording.
