# WP1.5 Readiness Risk Report

Date: 2026-03-12
Scope: WP1.5 readiness for handoff into WP2 (RL slot integration)
Decision: NOT READY (high-priority blockers present)

## Update (2026-03-14, WP1.5 Patch B)

- Runtime tray pose source now uses a dedicated name-preserving path:
  `/world/empty/dynamic_pose/info` -> `/tray_tracking/pose_stream_raw` -> `tray_pose_extractor_node` -> `/tray1/pose`.
- Legacy TF fallback path (`/world/empty/pose/info` -> `tray_pose_adapter_node`) remains available but is disabled by default.
- Expected impact: `/tray1/pose` no longer depends on name-less TF candidate fallback for tray selection.
- Validation commands:
  - `cd external/ENPM662_Group4_FinalProject/src/kitchen_robot_controller && PYTHONPATH=. pytest -q test/test_tray_pose_extractor_logic.py`
  - `cd external/ENPM662_Group4_FinalProject/src && source install/setup.bash && ros2 launch kitchen_robot_description gazebo.launch.py enable_legacy_tray_pose_adapter:=false headless:=true`
  - `source external/ENPM662_Group4_FinalProject/src/install/setup.bash && ros2 topic echo /tray1/pose --once`
  - `source external/ENPM662_Group4_FinalProject/src/install/setup.bash && ros2 node list | rg 'tray_pose_extractor|tray_pose_adapter'` (expect only `tray_pose_extractor` by default)

## Risk Register

### 1) Runtime substrate instability inherited from WP0 blocks reliable WP1.5 rollout generation
- Risk: WP1.5 depends on a stable runtime substrate (camera/TF/sync/latency). Current WP0 health evidence is still `FAIL/BLOCKED`.
- Severity: Critical
- Evidence:
  - `artifacts/wp0/wp0_report_after_bridge.json` reports `overall.result = FAIL` with counts `{'BLOCKED': 3, 'FAIL': 3, 'PASS': 0}`.
  - Failing sections include `camera_contract`, `tf_contract`, and `approx_sync`; blocked sections include `rosbag_replay`, `state_latency`, and `tray_stability`.
  - Example failure summaries in report: `Camera contract live validation failed.`, `TF contract runtime check failed or frames.pdf missing.`, `Approx sync requirement failed.`
- Impact: Rollout artifacts generated under unstable perception/timing can be low-quality or non-reproducible, weakening WP2 warm-start and benchmark validity.
- Validation Status: Failed
- Mitigation Options:
  - Close WP0 failing gates first (camera live contract, TF runtime/frames.pdf generation, approx sync success rate).
  - Unblock state latency and rosbag replay checks and make them mandatory preconditions for WP1.5 artifact export.

### 2) Current WP0 healthcheck unit-test regression indicates tooling contract drift
- Risk: A failing test in healthcheck tooling suggests command-contract behavior changed without corresponding test/tool alignment.
- Severity: High
- Evidence:
  - Command: `cd hrl_ws/src/hrl_trainer && PYTHONPATH=. pytest -q tests/test_v5_wp0_healthcheck_report.py tests/test_v5_wp0_metrics.py`
  - Result: `1 failed, 16 passed`
  - Failure snippet: `AssertionError: 0 != 1` in `test_run_rosbag_replay_image_diag_uses_single_remap_flag_with_all_rules` (expects one `--remap`, observed zero).
- Impact: Diagnostic and validation automation becomes less trustworthy; readiness signals can be noisy or wrong.
- Validation Status: Failed
- Mitigation Options:
  - Fix the command assembly logic or adjust test expectation if contract intentionally changed.
  - Add a regression test that validates full replay command tokens, not only count-based checks.

### 3) WP1.5 deliverables are only partially implemented versus stated exit criteria
- Risk: Key WP1.5 tasks in the implementation plan remain unchecked, especially rollout generator/replay integrity/curriculum/artifact export.
- Severity: High
- Evidence:
  - In `docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md` (WP1.5 section), these tasks remain unchecked:
    - rule-based rollout generator
    - deterministic replay + rollout integrity checks
    - curriculum YAMLs (`easy`, `medium`, `hard`)
    - training artifact export (rollout JSON/CSV + reward breakdown + canonical trajectories)
- Impact: WP2 cannot reliably consume standardized warm-start data; reproducibility and comparability across experiments are weakened.
- Validation Status: Not validated
- Mitigation Options:
  - Implement and test rollout generator + integrity checks first.
  - Add curriculum fixtures with fixed seeds and CI validation.
  - Define artifact schema/version and produce golden samples.

### 4) Validation depth is concentrated on unit contracts, not end-to-end WP1.5 flow
- Risk: Existing passing tests confirm schema and local logic, but not full data-path behavior across rollout generation and artifact lifecycle.
- Severity: Medium
- Evidence:
  - Passing unit/contract test runs:
    - `cd hrl_ws/src/hrl_trainer && PYTHONPATH=. pytest -q tests/test_v5_wp1_intent_layer.py tests/test_v5_wp1_acceptance.py tests/test_v5_wp1_5_rl_contracts.py`
    - Result: `16 passed in 0.05s`
  - WP1 acceptance harness passes (`scripts/v5/run_wp1_acceptance.sh`) with `success_count=30`, `fail_count=0`.
  - No corresponding evidence of completed end-to-end tests for rollout generation, replay integrity, and artifact reuse in WP2 handoff.
- Impact: Integration defects may remain hidden until WP2 training/evaluation, increasing rework cost and schedule risk.
- Validation Status: Partial
- Mitigation Options:
  - Add E2E tests covering `L1 -> observation/action contracts -> reward composer -> rollout artifact`.
  - Add deterministic replay checks that fail on drift.

### 5) Environment-dependent test invocation introduces reproducibility risk
- Risk: Tests fail at collection unless `PYTHONPATH` is explicitly configured.
- Severity: Medium
- Evidence:
  - Initial run without PYTHONPATH failed during collection with `ModuleNotFoundError: No module named 'hrl_trainer'`.
  - Re-run with `PYTHONPATH=.` succeeded for WP1/WP1.5 tests.
- Impact: Inconsistent local/CI behavior can mask regressions and slow incident triage.
- Validation Status: Confirmed
- Mitigation Options:
  - Standardize invocation via one project-level test entrypoint script.
  - Ensure package install/editable install or pytest config resolves imports consistently.

## Prioritized Remediation List

1. Fix WP0 hard failures and unblock blocked checks before declaring WP1.5 runtime-ready (camera contract, TF runtime/frames artifact, approx sync, state latency, rosbag replay).
   - Remediation B (runtime consistency): run `scripts/v5/run_wp1_5_runtime_parity_check.sh --mode both` and require `overall.result=PASS` to confirm startup parity across manual and auto-launch paths for `/clock`, `/joint_states`, `/v5/cam/overhead/rgb`, `/v5/cam/side/rgb`.
2. Resolve the failing healthcheck unit test (`--remap` contract drift) and lock replay command semantics with stronger regression coverage.
3. Implement missing WP1.5 core deliverables: rollout generator, deterministic replay/integrity checks, curriculum YAMLs, and artifact export pipeline.
4. Add WP1.5 E2E readiness tests spanning artifact generation and replay determinism across `easy/medium/hard` fixed seeds.
5. Remove PYTHONPATH fragility by standardizing test/bootstrap tooling for local and CI parity.

## Test Evidence Snippets (Current)

```text
$ cd hrl_ws/src/hrl_trainer && PYTHONPATH=. pytest -q tests/test_v5_wp1_intent_layer.py tests/test_v5_wp1_acceptance.py tests/test_v5_wp1_5_rl_contracts.py
................                                                         [100%]
16 passed in 0.05s
```

```text
$ scripts/v5/run_wp1_acceptance.sh
summary:
  success_count=30
  fail_count=0
  fail_reason_breakdown={}
```

```text
$ cd hrl_ws/src/hrl_trainer && PYTHONPATH=. pytest -q tests/test_v5_wp0_healthcheck_report.py tests/test_v5_wp0_metrics.py
..........F......                                                        [100%]
FAILED ... test_run_rosbag_replay_image_diag_uses_single_remap_flag_with_all_rules
E   AssertionError: 0 != 1
1 failed, 16 passed in 0.09s
```

```text
$ python3 - <<'PY' ... artifacts/wp0/wp0_report_after_bridge.json ...
overall {'counts': {'BLOCKED': 3, 'FAIL': 3, 'PASS': 0}, 'pass': False, 'result': 'FAIL'}
SECTION camera_contract status FAIL summary Camera contract live validation failed.
SECTION tf_contract status FAIL summary TF contract runtime check failed or frames.pdf missing.
SECTION approx_sync status FAIL summary Approx sync requirement failed.
SECTION rosbag_replay status BLOCKED summary Rosbag record/replay validation incomplete (replay not executed or samples unavailable).
SECTION state_latency status BLOCKED summary State topic latency validation unavailable.
SECTION tray_stability status BLOCKED summary Tray stability validation incomplete (missing live/jsonl inputs).
```

## Readiness Verdict

WP1.5 has strong contract-level progress but is not ready for WP2 handoff yet. Primary blockers are runtime health instability and incomplete WP1.5 rollout/replay/curriculum/artifact deliverables.
