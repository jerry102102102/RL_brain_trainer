# V5.1 TASK BOARD（Executable Tickets）

> 目的：把 V5.1 工作拆成可直接執行與可驗收的任務票，並明確切換到 `v5_1` 命名空間。

## T1 — Contract Freeze（L1/L2/L3 契約凍結）
- Goal: 凍結 V5.1 的資料契約與 topic/頻率邊界，避免實作階段契約漂移。
- Scope: 定義 L1/L2/L3 message schema、topic、rate、stale timeout、版本號；輸出 freeze 文檔與 schema 檔。
- DoD: Freeze 文檔與 schema 進 repo；所有 consumer/producer 能通過 schema check。
- Validation: `python scripts/v5_1/validate_contracts.py --strict` PASS。
- Artifacts: `artifacts/v5_1/contracts/contract_freeze_v1.json`, `artifacts/v5_1/contracts/compat_report.md`
- Owner: RL platform (L2/L3)
- Priority: P0
- Dependencies: 無
- Rollback: 回退至上一版 schema tag（`v5_1-contract-pre-freeze`），並暫停新 producer merge。

## T2 — L3 Executor（Deterministic Executor 主線化）
- Goal: 在 `v5_1` 命名空間落地 deterministic L3 executor。
- Scope: 建立 `hrl_trainer/v5_1/l3_executor/`，包含 command ingestion、interpolation、safety clamp、controller output。
- DoD: L3 在固定 seed/輸入下輸出 deterministic；對接 arm controller 成功。
- Validation: `pytest -q tests/v5_1/test_l3_executor_deterministic.py` PASS。
- Artifacts: `artifacts/v5_1/l3_executor/determinism_report.json`, `artifacts/v5_1/l3_executor/run.log`
- Owner: Control runtime
- Priority: P0
- Dependencies: T1
- Rollback: 切回 v5 相容 executor（唯讀維護路徑），保留接口不變。

## T3 — 安全與 Watchdog（Safety Shield + Runtime Watchdog）
- Goal: 建立 runtime 安全護欄與 watchdog，防止超限輸出與無回應狀態。
- Scope: limit checker、E-stop hook、heartbeat watchdog、timeout fallback。
- DoD: 觸發超限/timeout 時能 deterministic fallback 且有事件紀錄。
- Validation: `python scripts/v5_1/run_watchdog_fault_injection.py` PASS（含 timeout/limit 兩類）。
- Artifacts: `artifacts/v5_1/safety_watchdog/fault_injection_report.md`, `artifacts/v5_1/safety_watchdog/events.jsonl`
- Owner: Safety runtime
- Priority: P0
- Dependencies: T1, T2
- Rollback: 關閉新 watchdog 路徑，啟用 legacy safe-stop；保留 fault log。

## T4 — SAC Baseline（L2 Joint-space SAC 基線）
- Goal: 建立可重跑的 L2 joint-space SAC baseline。
- Scope: 訓練 config、seed 固定、評估腳本、基線指標（成功率/回報/違規率）。
- DoD: 至少 3 seeds 可重跑，指標落在基線窗口內。
- Validation: `bash scripts/v5_1/run_sac_baseline.sh --seeds 3` PASS。
- Artifacts: `artifacts/v5_1/l2_sac/baseline_metrics.csv`, `artifacts/v5_1/l2_sac/train_{seed}.log`
- Owner: RL training
- Priority: P1
- Dependencies: T1
- Rollback: 使用 rule-based L2 fallback baseline，阻擋新 policy promotion。

## T5 — Curriculum（課程式訓練路徑）
- Goal: 建立由簡到難的 curriculum，穩定提升 L2 policy。
- Scope: stage 定義、promotion gate、回退規則、資料記錄。
- DoD: 每 stage 有明確 gate；promotion/rollback 事件可追蹤。
- Validation: `python scripts/v5_1/run_curriculum.py --check-gates` PASS。
- Artifacts: `artifacts/v5_1/curriculum/stage_summary.json`, `artifacts/v5_1/curriculum/promotion_log.jsonl`
- Owner: RL training
- Priority: P1
- Dependencies: T4
- Rollback: 回退到前一 stage checkpoint 並鎖定 promotion。

## T6 — E2E Pipeline（端到端管線）
- Goal: 打通 L1→L2→L3 E2E pipeline，且每次執行都包含環境 bring-up 驗證。
- Scope: bring-up script、health check、E2E smoke run、artifact 落盤。
- DoD: 一鍵命令可完成 bring-up + run + basic checks。
- Validation: `bash scripts/v5_1/run_e2e_pipeline.sh --bringup --verify` PASS。
- Artifacts: `artifacts/v5_1/e2e/{run_id}/pipeline.log`, `artifacts/v5_1/e2e/{run_id}/healthcheck.json`
- Owner: Integration
- Priority: P0
- Dependencies: T2, T3, T4
- Rollback: 禁止宣稱 E2E PASS；改回分層單測模式。

## T7 — Layer Logging（L1/L2/L3 分層記錄）
- Goal: 每層輸出最小欄位固定化，便於故障定位與回放。
- Scope: 統一 log schema（L1/L2/L3）、run_id/session_id、timestamp、input/output、decision reason、latency。
- DoD: 每次 pipeline 都產生三層 log，且可由工具聚合。
- Validation: `python scripts/v5_1/validate_layer_logs.py --run-latest` PASS。
- Artifacts: `artifacts/v5_1/logs/l1/*.jsonl`, `artifacts/v5_1/logs/l2/*.jsonl`, `artifacts/v5_1/logs/l3/*.jsonl`
- Owner: Observability
- Priority: P0
- Dependencies: T1, T6
- Rollback: 切回最小 stderr logging（僅緊急），並標記 run 為不可驗證。

## T8 — Gates 驗收（Release Gate for V5.1）
- Goal: 定義可執行的 V5.1 驗收 gate，避免「看起來可用」但不可重跑。
- Scope: 合約 gate、determinism gate、安全 gate、E2E gate、log 完整性 gate。
- DoD: 所有 gate 有明確 PASS/FAIL 與 blocking 規則；形成驗收報告模板。
- Validation: `python scripts/v5_1/run_release_gates.py --profile v5_1` PASS。
- Artifacts: `artifacts/v5_1/gates/gate_report.md`, `artifacts/v5_1/gates/gate_results.json`
- Owner: Tech lead + QA
- Priority: P0
- Dependencies: T1~T7
- Rollback: 驗收 fail 時不允許升版 tag；回到對應失敗票處理。

---

## 執行順序建議
1. P0 先行：T1 → T2 → T3 → T6 → T7 → T8
2. P1 併行：T4 → T5（在 T1 完成後展開）
