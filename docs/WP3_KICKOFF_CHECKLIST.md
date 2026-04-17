# WP3 Kickoff Checklist (Jerry one-page)

> Scope: 先把 WP3 跑起來（HIL gate + 統計信心 + safety/rollback），不改核心演算法。

## A. 開工前 10 分鐘
- [ ] 確認 branch 與目標 repo 正確
  ```bash
  git rev-parse --abbrev-ref HEAD && pwd
  ```
- [ ] 建立 WP3 目錄骨架
  ```bash
  mkdir -p docs/wp3 scripts/wp3 artifacts/wp3/{hil_dryrun,stat_report,failure_rehearsal,gate_runs}
  ```
- [ ] 讀過主計劃並確認 DoD
  ```bash
  test -f docs/WP3_EXECUTION_PLAN_2026-03-22.md && echo "Plan ready"
  ```

## B. WS1（HIL/runtime gate）
- [ ] runtime contract 完成：`docs/wp3/ws1_runtime_contract.md`
- [ ] HIL profile + checklist 完成：`configs/hil_runtime_profile.yaml`, `docs/wp3/hil_env_checklist.md`
- [ ] wrapper 檢查通過
  ```bash
  bash -n scripts/wp3/run_hil_gate.sh
  ```
- [ ] 至少一次 dry-run，產生 `artifacts/wp3/hil_dryrun/<date>/`

## C. WS2（seed/episodes + 統計信心）
- [ ] matrix 文件完成：`docs/wp3/ws2_experiment_matrix.md`
- [ ] batch script 檢查通過
  ```bash
  bash -n scripts/wp3/run_seed_episode_matrix.sh
  ```
- [ ] confidence policy 完成：`docs/wp3/ws2_confidence_policy.md`
- [ ] 產生統計摘要（含 CI/variance）

## D. WS3（safety gate + rollback）
- [ ] safety gate 規格完成：`docs/wp3/ws3_safety_gate.md`
- [ ] rollback spec 完成：`docs/wp3/ws3_rollback_spec.md`
- [ ] runner 檢查通過
  ```bash
  bash -n scripts/wp3/run_safety_gate_and_rollback.sh
  ```
- [ ] failure rehearsal 有 evidence：`artifacts/wp3/failure_rehearsal/<date>/`

## E. WS4（整合與決策）
- [ ] full gate script 檢查通過
  ```bash
  bash -n scripts/wp3/run_wp3_full_gate.sh
  ```
- [ ] 產生 gate runs：`artifacts/wp3/gate_runs/`
- [ ] Go/No-Go packet 完成：`docs/wp3/WP3_GO_NO_GO_PACKET.md`

## Done 判準（最終勾選）
- [ ] real path 與 real robot runtime 術語與證據已分開呈現
- [ ] HIL/runtime gate、stat confidence gate、safety+rollback gate 全部有可追溯證據
- [ ] Jerry 可用一條命令鏈重跑 WP3 gate
