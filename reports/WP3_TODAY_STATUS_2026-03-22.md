# WP3 Today Status (2026-03-22)

## Done
1. 建立 today-only 執行清單：
   - `docs/WP3_TODAY_EXECUTION_2026-03-22.md`
2. 完成一輪 WP3 preflight + gate evidence（dry-run/document check 層級）：
   - preflight 環境快照
   - runtime/HIL contract 檢查
   - 統計 gate 最小檢查
   - rollback gate 檢查

## Not done
1. 實機/HIL runtime gate 實跑（無法聲稱完成，今日僅完成 dry-run/document evidence）
2. 統計 confidence gate（CI/variance + threshold 判定）完整判讀尚未完成
3. rollback automation 真實 failure injection rehearsal 尚未完成

## Blockers
1. `docs/wp3/ws1_runtime_contract.md` 缺失
2. `configs/hil_runtime_profile.yaml` 缺失
3. `docs/wp3/hil_env_checklist.md` 缺失
4. `docs/wp3/ws3_safety_gate.md`、`docs/wp3/ws3_rollback_spec.md` 缺失
5. `scripts/wp3/run_safety_gate_and_rollback.sh` 缺失
6. 無今日實機/HIL slot 證據（僅能做 dry-run）

## Evidence
- Preflight：
  - 命令：
    ```bash
    {
      echo "timestamp=$(date -Iseconds)";
      echo "branch=$(git rev-parse --abbrev-ref HEAD)";
      echo "pwd=$(pwd)";
    } | tee artifacts/wp3/gate_runs/2026-03-22/preflight_env.txt
    ```
  - 證據：`artifacts/wp3/gate_runs/2026-03-22/preflight_env.txt`

- Runtime/HIL contract 檢查：
  - 命令：
    ```bash
    {
      test -f docs/wp3/ws1_runtime_contract.md && echo found || echo missing;
      test -f configs/hil_runtime_profile.yaml && echo found || echo missing;
      test -f docs/wp3/hil_env_checklist.md && echo found || echo missing;
    } | tee artifacts/wp3/gate_runs/2026-03-22/runtime_hil_contract_check.log
    ```
  - 結果：FAIL（缺檔）
  - 證據：`artifacts/wp3/gate_runs/2026-03-22/runtime_hil_contract_check.log`

- 統計 gate 最小檢查：
  - 命令：
    ```bash
    python3 - <<'PY' | tee artifacts/wp3/gate_runs/2026-03-22/stat_gate_min_check.log
    import json
    p='artifacts/reports/v5/m2_8_formal_comparison_summary_seed42_ep8.json'
    obj=json.load(open(p))
    print('keys', sorted(obj.keys()))
    PY
    ```
  - 結果：PASS（baseline summary 可讀取）；但僅是「最小檢查」，非完整 CI/variance gate
  - 證據：`artifacts/wp3/gate_runs/2026-03-22/stat_gate_min_check.log`

- Rollback gate 檢查：
  - 命令：
    ```bash
    {
      test -f docs/wp3/ws3_rollback_spec.md || echo missing;
      test -f scripts/wp3/run_safety_gate_and_rollback.sh || echo missing;
    } | tee artifacts/wp3/gate_runs/2026-03-22/rollback_gate_check.log
    ```
  - 結果：FAIL（spec/script 缺失）
  - 證據：`artifacts/wp3/gate_runs/2026-03-22/rollback_gate_check.log`

## Go-or-NoGo
- **NoGo（today）**
- 理由：runtime/HIL contract 與 rollback gate 核心文件/腳本缺失，且無實機/HIL runtime 證據。

## 最小下一步（30-60 分鐘內可執行）
1. 先補齊三個最小文件骨架（30 分鐘）：
   - `docs/wp3/ws1_runtime_contract.md`
   - `configs/hil_runtime_profile.yaml`
   - `docs/wp3/ws3_rollback_spec.md`
2. 補一個 rollback runner stub + parse check（15-20 分鐘）：
   - `scripts/wp3/run_safety_gate_and_rollback.sh`
   - `bash -n scripts/wp3/run_safety_gate_and_rollback.sh`
3. 重跑三個 gate check（10 分鐘）並更新同日 evidence logs。
