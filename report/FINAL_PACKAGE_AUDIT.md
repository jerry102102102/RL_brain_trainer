# Final Package Audit

Purpose: Audit the current repository state before building the final ENPM690 report package.

## Existing Report / Documentation Files

The current repo stores current docs in `docs/`, archived research notes in `docs/archive/`, and final-package deliverables in `report/`.

- `docs/README.md`
- `docs/CURRENT_IMPLEMENTATION.md`
- `docs/PHASE1_APPROACH_DOCK_CLOSEOUT.md`
- `docs/PHASE2_FINAL_DEMONSTRATION_REPORT.md`
- `docs/PHASE3A_GZ_MIGRATION_STATUS.md`
- `docs/PHASE3A_GZ_REUSE_AUDIT.md`
- `docs/QUICKSTART.md`
- `docs/REPO_CLEANUP_MAIN_PREP.md`
- `docs/RL_WORKSPACE_AND_TRANSPORT_STATUS.md`
- `docs/ROUTE_CURRICULUM_TRAINING_PLAN.md`
- `docs/V5_QWEN_MCP_BRIDGE.md`

Legacy planning and diagnostic docs were moved to:

- `docs/archive/final_cleanup_20260513/`

## Official Artifacts Found

- `docs/PHASE2_FINAL_DEMONSTRATION_REPORT.md`
- `docs/RL_WORKSPACE_AND_TRANSPORT_STATUS.md`
- `docs/ROUTE_CURRICULUM_TRAINING_PLAN.md`
- `artifacts/kinematic_phase1/phase1c/workspace_sweep_workspace_noop_vs_previous_summary_001.json`
- `artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json`
- `artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip`
- `artifacts/kinematic_phase1/route_curriculum/prefix120_teacher_anchor/teacher_route_anchor_dataset.npz`
- `artifacts/kinematic_phase1/route_curriculum/route_segment121_180_teacheranchored_smoke_001/route_gate_full483/route_gate_summary.json`

## Missing Or Optional Artifacts

- None for required final-package artifacts.

## Files To Regenerate For Final Package

- `report/FINAL_REPORT.md`
- `report/FINAL_REPORT.pdf`
- `report/FINAL_PRESENTATION.pptx`
- `report/figures/*.png`
- `report/demo_outputs/*.json` (local/generated; ignored for main unless explicitly promoted)

## Artifacts Not To Use As Official Best

- `artifacts/kinematic_phase1/route_curriculum/route_prefix180_routeobs_sequence2_1m_001/model_latest.zip`
- `artifacts/kinematic_phase1/route_curriculum/route_prefix180_routeobs_sequence2_antiforget_1m_001/model_latest.zip`
- `artifacts/kinematic_phase1/route_curriculum/route_segment121_180_teacheranchored_1m_001/model_latest.zip`

These are useful negative-result artifacts, but they fail sequential retention and should not replace the official prefix120 checkpoint.

## Official Numbers For Final Package

- Stage 5 kinematic skill success: `0.93`
- Stage 5 final position error: `2.89 mm`
- Stage 5 final orientation error: `0.0208 rad`
- Route baseline full483 success: `0.0435`
- Route baseline longest prefix: `21`
- Route prefix120 success: `1.0`
- Route full483 probe success: `0.4741`
- Route full483 probe longest prefix: `170`
- Random-start known-workspace success: `0.802`
- Random-start frontier success: `0.240`
- Random-start full-stress success: `0.219`

## Audit Conclusion

The final package can be generated from existing repo artifacts and documentation. The package must state clearly that full holder1-to-holder8 Gazebo transport is not solved.
