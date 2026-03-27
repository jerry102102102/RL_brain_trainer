# V5.1 T4~T6 Delivery Report

## Scope
- T4 Curriculum（S0/S1/S2 漸進）
- T5 Gates 驗收機制（最小可執行）
- T6 E2E pipeline（串 L1/L2/L3 + logs + artifact）

## Changed Files
### Code
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/curriculum.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/gates.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/pipeline_e2e.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/__init__.py`

### Tests
- `hrl_ws/src/hrl_trainer/tests/test_v5_1_curriculum.py`
- `hrl_ws/src/hrl_trainer/tests/test_v5_1_gates.py`
- `hrl_ws/src/hrl_trainer/tests/test_v5_1_pipeline_e2e.py`

## Implementation Summary
### T4 Curriculum
- Added deterministic `CurriculumManager` with staged progression:
  - `S0 -> S1 -> S2`
  - stage-local counters + promotion criteria (`min_episodes`, `promote_success_rate`)
- Added artifact export API `to_artifact()` for inspectable state/history.

### T5 Gates
- Added minimal executable gate evaluator:
  - `GateSpec` + `GateEvaluator`
  - checks `episodes`, `success_rate`, `intervention_rate`
  - emits pass/fail + reasons + normalized score
- Added `write_gate_report()` JSON artifact output.

### T6 E2E Pipeline
- Added `run_pipeline_e2e()` and CLI module entry in `pipeline_e2e.py`.
- Pipeline composes:
  - existing `run_smoke()` for L1/L2/L3 layered JSONL logs
  - curriculum progression per episode
  - gate evaluation at run end
- Emits inspectable artifacts:
  - `pipeline_summary.json`
  - `curriculum_state.json`
  - `gate_result.json`
  - layered logs under `logs/l1`, `logs/l2`, `logs/l3`

## Validation Commands & Results
### Unit tests (required reproducible with `/usr/bin/python3`)
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer
/usr/bin/python3 -m pytest -q tests/test_v5_1_curriculum.py tests/test_v5_1_gates.py tests/test_v5_1_pipeline_e2e.py
```
Result: **PASS (6 passed)**

### Local E2E smoke
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer
/usr/bin/python3 -m hrl_trainer.v5_1.pipeline_e2e --run-id subagent_t4_t6 --episodes 4 --steps-per-episode 3 --artifact-root artifacts/v5_1/e2e/subagent_t4_t6
```
Result: **PASS** (artifacts/logs generated)

## Artifact Paths (generated)
- `hrl_ws/src/hrl_trainer/artifacts/v5_1/e2e/subagent_t4_t6/pipeline_summary.json`
- `hrl_ws/src/hrl_trainer/artifacts/v5_1/e2e/subagent_t4_t6/curriculum_state.json`
- `hrl_ws/src/hrl_trainer/artifacts/v5_1/e2e/subagent_t4_t6/gate_result.json`
- `hrl_ws/src/hrl_trainer/artifacts/v5_1/e2e/subagent_t4_t6/logs/l1/*.jsonl`
- `hrl_ws/src/hrl_trainer/artifacts/v5_1/e2e/subagent_t4_t6/logs/l2/*.jsonl`
- `hrl_ws/src/hrl_trainer/artifacts/v5_1/e2e/subagent_t4_t6/logs/l3/*.jsonl`

## Constraints Check
- [x] 新功能僅放在 `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/`
- [x] 未新增 task1 路徑功能
- [x] 測試可由 `/usr/bin/python3` 重跑
- [x] 產出可檢查 artifacts 與分層 logs

## Risks / Limits
- Current E2E success/intervention metrics are deterministic smoke heuristics (for local CI reproducibility), not real robot performance metrics.
- Gate thresholds are minimal executable defaults; may need alignment with final acceptance criteria from system-level runs.

## Next Step
- Main can re-run the same test/smoke commands above on target environment, then optionally wire real runtime metrics into `pipeline_e2e.py` gate inputs.
