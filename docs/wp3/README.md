# WP3 Gates - Implementation Entry Points

## One-command runner
```bash
bash scripts/wp3_run_gates.sh --seeds 11,13,17 --episodes 4 --rollback-episodes 8 --rollback-seed 42
```

若要先自動產生 WS1 HIL dry-run evidence 再跑 gates：
```bash
bash scripts/wp3_run_gates.sh \
  --with-hil-dryrun \
  --hil-mode mock \
  --hil-policy rule_l2_v0 \
  --hil-seed 42 \
  --seeds 11,13,17 --episodes 4 --rollback-episodes 8 --rollback-seed 42
```

## Individual gates

### WS1 Runtime/HIL gate
先產生 evidence（dry-run/mock）：
```bash
bash scripts/wp3/run_hil_dryrun_and_capture.sh --mode mock --policy rule_l2_v0 --seed 42
```
Evidence output:
- `artifacts/wp3/hil_dryrun/<timestamp>/hil_runtime_evidence.json`
- `artifacts/wp3/hil_dryrun/<timestamp>/hil_runtime_evidence.log`

再跑 WS1 gate：
```bash
bash scripts/wp3/run_hil_gate.sh
```
Gate evidence: `artifacts/reports/wp3/<timestamp>/ws1_runtime_hil_gate.json`

### WS2 Seed sweep statistical gate
```bash
bash scripts/wp3/run_seed_episode_matrix.sh --seeds 11,13,17 --episodes 4 --policy rl_l2
```
Evidence:
- `artifacts/reports/wp3/<timestamp>/ws2_seed_sweep_summary.json`
- `artifacts/reports/wp3/<timestamp>/ws2_per_seed/seed_<seed>_summary.json`

Summary includes mean/std/95% CI half width for success rate and average reward.

### WS3 Safety/Rollback gate
```bash
bash scripts/wp3/run_safety_gate_and_rollback.sh --episodes 8 --seed 42
```
Evidence: `artifacts/reports/wp3/<timestamp>/ws3_rollback_gate.json`

Rollback gate checks at least:
- execution can be forced to `rule_l2_v0`
- strict policy path (no fallback)
- eval harness pass status

## HIL evidence JSON schema (WS1)
Schema file: `docs/wp3/hil_runtime_evidence.schema.json`

Required fields:
- `timestamp`
- `runtime_source`
- `policy`
- `seed`
- `checks.health.ok`
- `checks.topic.ok`
- `checks.bridge.ok`
- `pass`
- `notes`

WS1 PASS needs:
1. real path evidence exists (`artifacts/reports/v5/m2_8_formal_comparison_summary_seed42_ep8.json`)
2. latest `hil_runtime_evidence.json` exists under `artifacts/wp3/hil_dryrun`
3. evidence fields/schema are valid
4. `pass=true`
5. health/topic/bridge checks all `ok=true`

## Notes
- `run_hil_dryrun_and_capture.sh` 的預設模式是 `mock`，屬於 pipeline 驗證證據，不代表真實機器人 motion。
- 若要宣稱 real HIL，請以真機 runtime source 與真實 logs 取代 mock evidence。
