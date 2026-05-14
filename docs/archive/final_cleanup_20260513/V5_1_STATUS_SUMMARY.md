# V5.1 Status Summary（短版）

- 日期：2026-03-27
- 2026-03-27 晚間更新：V5.1 已收斂為單一路徑（ROS2 + venv + SAC Torch）
- 範圍：RBT（RL_brain_trainer）V5.1 方向收斂與可交付狀態
- 參考主文檔：`docs/V5_1_IMPLEMENTATION_PLAN.md`

---

## 1) 目前完成（Done）

1. **方向凍結**
   - V5.1 已明確採用：**L1 perception+decision / L2 joint-space RL（SAC Torch） / L3 deterministic execution+safety**。
   - 相容路線已移除：不再保留 numpy SAC / rule policy mode。
   - 主要契約與實驗門檻已在 `docs/V5_1_IMPLEMENTATION_PLAN.md` 定義。

2. **前置工程基礎可用（沿用 V5/WP1.5）**
   - WP1.5 runtime hotfix 與 parity 檢查文件已具備。
   - 相關風險與檢查入口已存在：
     - `docs/WP1_5_RUNTIME_PARITY_CHECKER.md`
     - `docs/WP1_5_READINESS_RISK_REPORT.md`

3. **關鍵 rerun 證據已留存（報告層）**
   - `reports/task1_real_l3_rerun_report_2026-03-22.md`

---

## 2) 目前未完成（Open）

1. **V5.1 票務級實作尚未完整落地**
   - L2 obs/action schema validator（Ticket 2.1）
   - SAC baseline trainer 完整流程（Ticket 2.2）
   - L3 safety pipeline 五件套標準化落地（Ticket 2.3）
   - curriculum engine（Ticket 2.4）
   - experiment harness（Exp-A/B/C）與自動報告（Ticket 2.5）
   - rollback tooling（Ticket 2.6）

2. **V5.1 的一鍵腳本仍屬草案**
   - 計劃書中的 `run_v5_1_baseline.sh / run_v5_1_experiment.sh / rerun_v5_1_all.sh` 尚需實作與驗證。

3. **成功門檻尚未形成最終 artifact 套件**
   - Exp-A/B/C 的可審核結果（含跨 seed 統計與 gate decision）尚未完整產出。

---

## 3) 下一步（Next）

1. **先凍結契約與配置（優先）**
   - 先完成 Ticket 2.1（schema + validator + tests），避免接口漂移。

2. **建立最小可重跑訓練主幹**
   - 依 Ticket 2.2/2.3 完成 SAC baseline + L3 safety pipeline。

3. **接上 curriculum + 三個優先實驗**
   - 完成 Ticket 2.4/2.5，固定 Exp-A/B/C 輸出格式與門檻判定。

4. **補齊回滾能力與 DoD 打包**
   - 完成 Ticket 2.6，確保 checkpoint/config/commit/report 可對齊回退。

---

## 4) 建議審核順序

1. `docs/V5_1_STATUS_SUMMARY.md`（本文件）
2. `docs/V5_1_IMPLEMENTATION_PLAN.md`
3. `docs/WP1_5_RUNTIME_PARITY_CHECKER.md`
4. `docs/WP1_5_READINESS_RISK_REPORT.md`
5. `reports/task1_real_l3_rerun_report_2026-03-22.md`
