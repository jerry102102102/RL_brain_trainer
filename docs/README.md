# Documentation Index

This folder contains both current documentation and legacy research notes. For main-branch review, treat the files below as the canonical docs.

## Current Docs

- `docs/CURRENT_IMPLEMENTATION.md`: current implementation truth and limitations.
- `docs/QUICKSTART.md`: quick setup and common commands.
- `docs/PHASE1_APPROACH_DOCK_CLOSEOUT.md`: Phase 1 kinematic closeout.
- `docs/PHASE2_FINAL_DEMONSTRATION_REPORT.md`: Phase 2 final demonstration report source.
- `docs/V5_QWEN_MCP_BRIDGE.md`: L1 Qwen/MCP semantic bridge.
- `docs/PHASE3A_GZ_REUSE_AUDIT.md`: reusable Gazebo/ROS2 integration audit.
- `docs/PHASE3A_GZ_MIGRATION_STATUS.md`: Phase 3A Gazebo migration status.
- `docs/RL_WORKSPACE_AND_TRANSPORT_STATUS.md`: workspace and transport status.
- `docs/ROUTE_CURRICULUM_TRAINING_PLAN.md`: route curriculum design and status.

## Final Package

- `report/FINAL_PROJECT_SUMMARY.md`
- `report/OFFICIAL_ARTIFACTS.md`
- `report/FINAL_REPORT.md`
- `report/FINAL_REPORT.pdf`
- `report/FINAL_PRESENTATION.pptx`
- `report/DEMO_VIDEO_SCRIPT.md`
- `report/DEMO_RECORDING_COMMANDS.md`

## Legacy / Diagnostic Docs

Older V4, V5.1, WP0/WP1.5, memory, bridge, Dock-Coarse, IK, and planning documents are preserved under `docs/archive/` for traceability. They should not be used as the current project story unless a current doc explicitly links to them.

The current final story is:

```text
Qwen L1 semantic bridge
-> structured IntentPacket / skill request
-> kinematic Approach -> Finisher RL stack
-> workspace expansion and route-curriculum analysis
-> Gazebo/RViz demo path for visualization
```

Do not claim:

- full holder1 -> holder8 tray transport is solved,
- Stage 10/11 is the full workspace,
- Gazebo physics/contact validation is complete,
- real robot deployment is complete.
