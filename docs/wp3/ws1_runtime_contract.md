# WS1 Runtime/HIL Contract

WP3 WS1 gate 判準需同時具備：

1. **real path evidence**（repo/runtime 內可重現）
2. **runtime/HIL evidence JSON + log**（可為 dry-run/mock 或真實 HIL）

## Evidence path convention
`artifacts/wp3/hil_dryrun/<timestamp>/`

最低必備輸出：
- `hil_runtime_evidence.json`
- `hil_runtime_evidence.log`

## HIL runtime evidence schema (minimum)
參考：`docs/wp3/hil_runtime_evidence.schema.json`

必要欄位：
- `timestamp` (ISO8601 UTC)
- `runtime_source` (e.g. `simulated_hil_dryrun`, `real_robot_hil`)
- `policy`
- `seed`
- `checks.health.ok` (bool)
- `checks.topic.ok` (bool)
- `checks.bridge.ok` (bool)
- `pass` (bool)
- `notes`

## Gate checker behavior
`python -m hrl_trainer.v5.wp3_gates ws1` 會檢查：
- real path probe: `artifacts/reports/v5/m2_8_formal_comparison_summary_seed42_ep8.json`
- latest HIL evidence JSON: `artifacts/wp3/hil_dryrun/**/hil_runtime_evidence.json`
- schema minimum fields
- `pass=true`
- health/topic/bridge checks all `ok=true`

## Dry-run generator
```bash
bash scripts/wp3/run_hil_dryrun_and_capture.sh --mode mock --policy rule_l2_v0 --seed 42
```

> 注意：`--mode mock` 僅代表 pipeline 證據流程演練，不可當成已完成真實實機驗證。
