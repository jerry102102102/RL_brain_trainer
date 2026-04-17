# WP3 Today-only Execution (2026-03-22)

> 原則：今天只做可落地、可驗證項目；不重寫長排程。 
> 參照：`docs/WP3_EXECUTION_PLAN_2026-03-22.md`、`docs/WP3_KICKOFF_CHECKLIST.md`

## 1) Preflight 環境快照（必做）
- 命令：
```bash
{
  echo "timestamp=$(date -Iseconds)";
  echo "branch=$(git rev-parse --abbrev-ref HEAD)";
  echo "pwd=$(pwd)";
} | tee artifacts/wp3/gate_runs/2026-03-22/preflight_env.txt
```
- PASS 判準：`preflight_env.txt` 內含 `timestamp/branch/pwd` 三行
- FAIL 判準：檔案不存在或欄位缺失
- 輸出路徑：`artifacts/wp3/gate_runs/2026-03-22/preflight_env.txt`

## 2) Runtime/HIL contract 最小檢查（必做）
- 命令：
```bash
{
  test -f docs/wp3/ws1_runtime_contract.md || echo 'missing: docs/wp3/ws1_runtime_contract.md';
  test -f configs/hil_runtime_profile.yaml || echo 'missing: configs/hil_runtime_profile.yaml';
  test -f docs/wp3/hil_env_checklist.md || echo 'missing: docs/wp3/hil_env_checklist.md';
} | tee artifacts/wp3/gate_runs/2026-03-22/runtime_hil_contract_check.log
```
- PASS 判準：三個檔案皆存在，且可進入 HIL/real runtime 執行
- FAIL 判準：任一檔案缺失（視為 HIL blocker）
- 輸出路徑：`artifacts/wp3/gate_runs/2026-03-22/runtime_hil_contract_check.log`

## 3) 統計 gate 最小檢查（必做）
- 命令：
```bash
python3 - <<'PY' | tee artifacts/wp3/gate_runs/2026-03-22/stat_gate_min_check.log
import json
p='artifacts/reports/v5/m2_8_formal_comparison_summary_seed42_ep8.json'
obj=json.load(open(p))
print('found', p)
print('keys', sorted(obj.keys()))
PY
```
- PASS 判準：可讀取既有統計摘要檔（至少代表 baseline evidence 可被載入）
- FAIL 判準：檔案不存在或 JSON 無法解析
- 輸出路徑：`artifacts/wp3/gate_runs/2026-03-22/stat_gate_min_check.log`

## 4) Rollback gate 最小檢查（必做）
- 命令：
```bash
{
  test -f docs/wp3/ws3_rollback_spec.md || echo 'missing: docs/wp3/ws3_rollback_spec.md';
  test -f scripts/wp3/run_safety_gate_and_rollback.sh || echo 'missing: scripts/wp3/run_safety_gate_and_rollback.sh';
} | tee artifacts/wp3/gate_runs/2026-03-22/rollback_gate_check.log
```
- PASS 判準：rollback spec + runner script 都存在，且 runner 可 `bash -n`
- FAIL 判準：任一檔案缺失或語法檢查失敗
- 輸出路徑：`artifacts/wp3/gate_runs/2026-03-22/rollback_gate_check.log`

## 5) 今日結論報告（必做）
- 命令：
```bash
test -f reports/WP3_TODAY_STATUS_2026-03-22.md
```
- PASS 判準：報告存在，且含 Done / Not done / Blockers / Evidence / Go-or-NoGo
- FAIL 判準：報告不存在或缺段落
- 輸出路徑：`reports/WP3_TODAY_STATUS_2026-03-22.md`
