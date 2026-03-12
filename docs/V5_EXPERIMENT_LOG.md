# V5 Experiment Log

## 2026-03-11 - WP1 runtime slot map wiring and acceptance harness
- Added runtime slot map loading path in L1 intent layer using `hrl_ws/src/hrl_trainer/config/v5_slot_map.yaml`.
- Added WP1 acceptance pipeline harness with:
  - 10-task deterministic smoke run.
  - 20-task seeded random run.
  - Summary counters: `success_count`, `fail_count`, `fail_reason_breakdown`.
- Added acceptance runner script: `scripts/v5/run_wp1_acceptance.sh`.
- Added/updated tests:
  - `test_v5_wp1_intent_layer.py` runtime config load coverage.
  - `test_v5_wp1_acceptance.py` smoke/random summary coverage.

## 2026-03-12 - WP1.5 readiness risk report baseline
- Added `docs/WP1_5_READINESS_RISK_REPORT.md` with current readiness verdict, risk register, and test evidence snippets.
- Current snapshot:
  - WP1/WP1.5 contract tests pass (`16 passed`).
  - WP1 acceptance harness reports `success_count=30`, `fail_count=0`.
  - WP0 healthcheck tests include one failure (`AssertionError: 0 != 1` in remap contract test), and `artifacts/wp0/wp0_report_after_bridge.json` remains overall `FAIL`.
