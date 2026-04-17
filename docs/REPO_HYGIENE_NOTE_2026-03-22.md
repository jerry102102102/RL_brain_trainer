# Repo Hygiene Note — 2026-03-22

## 目標
降低 `git status --short` 的未追蹤噪音，保留真正應提交的程式/文件/核心報告，讓後續開發與 code review 更乾淨。

## 盤點結果（依 `git status --short`）

### A. 應該被版本控制（保留追蹤）
> 原則：功能程式碼、測試、設計文件、可重現研究結果（尤其 v5 報告）

1. **已追蹤且有修改（M）**
   - `README.md`
   - `docs/*.md`（多個 v5 / WP 文件）
   - `hrl_ws/src/hrl_trainer/...`（v5 模組與測試）
2. **未追蹤但高機率應納版（??）**
   - `hrl_ws/src/hrl_trainer/hrl_trainer/v5/*.py`（新功能模組）
   - `hrl_ws/src/hrl_trainer/tests/test_v5_*.py`（對應測試）
   - `docs/WP0_GAP_ANALYSIS.md`, `docs/WP2_IMPLEMENTATION_NOTE.md`
   - `artifacts/reports/v5/*`（核心評估結果，應保留）
   - `artifacts/wp1_5/runtime_parity_report*.json`（WP1.5 產出，建議保留）
   - `scripts/m2_9_rerun_wp2.sh`
   - `external/ENPM662_Group4_FinalProject/` 內的**原始碼與文件**（若此資料夾為 repo 正式內容）

### B. 應忽略（已加入 `.gitignore`）
> 原則：編譯輸出、快取、local runtime log、一次性診斷輸出、Graphviz frames

- Python 快取
  - `__pycache__/`, `*.py[cod]`
- 建置/安裝/執行日誌目錄
  - `build/`, `install/`, `log/`, `**/build/`, `**/install/`, `**/log/`
- Graphviz / frame dump
  - `frames_*.gv`, `frames_*.pdf`
  - `artifacts/wp0/frames*.gv`, `artifacts/wp0/frames*.pdf`
- WP0 在地診斷與 runtime 噪音
  - `artifacts/wp0/ros_logs/`, `artifacts/wp0/ros2_logs/`, `artifacts/wp0/smoke_runs/`
  - `artifacts/wp0/*_live*.json`, `*_after_*.json`, `replay_*.json`, `state_latency*.json`, `tf_check*.json`
  - `artifacts/wp0/id_switch.json`, `image_live.json`, `pose_jitter.json`
  - `artifacts/wp0/rosbag_print_commands.json`, `scene_colcon_build.log`, `scene_launch.log`
- 子專案/工作區快取
  - `external/**/__pycache__/`, `hrl_ws/**/__pycache__/`

### C. 可刪除的暫存/快取（僅建議，不在本次執行刪除）
> 這些多數已被 ignore，未來可定期清理磁碟

- 根目錄與子目錄內 `build/`, `install/`, `log/`
- `__pycache__/` 全域快取
- `frames_*.gv|pdf` 與 `artifacts/wp0/frames*`
- `artifacts/wp0/ros_logs/`, `artifacts/wp0/ros2_logs/`, `artifacts/wp0/smoke_runs/`

---

## 本次 `.gitignore` 調整重點
- 新增基礎忽略：Python cache + build/install/log
- 新增 Graphviz frames 忽略規則
- 新增 WP0 在地診斷噪音規則
- 新增 nested project `__pycache__` 規則
- **保護性白名單**（避免誤傷核心報告來源）：
  - `!artifacts/reports/`
  - `!artifacts/reports/v5/`
  - `!artifacts/reports/v5/**`
  - `!docs/reports/`
  - `!docs/reports/**`

---

## 驗證（前後比較）
- Before: `git status --short` 共 **162** 行（未追蹤 `??` = **147**）
- After : `git status --short` 共 **71** 行（未追蹤 `??` = **56**）
- 噪音下降：
  - 總行數減少 **91**
  - 未追蹤噪音減少 **91**

結論：本次已顯著降低「快取/日誌/一次性輸出」噪音；剩餘多數是**實質代碼、文件、報告候選**，可進入下一步人工篩選與分批 commit。

---

## 後續建議
1. **先確認 `external/ENPM662_Group4_FinalProject/` 的納版策略**
   - 若要納版：分批提交（docs/src 分開）
   - 若只作本地參考：考慮改用 `.git/info/exclude` 或明確目錄規則（需團隊共識）
2. **將成果與噪音分 commit**
   - Commit 1: `.gitignore` + hygiene note
   - Commit 2+: 功能程式碼與測試
   - Commit 3: 報告與文件
3. **定期檢查與清理**（本地）
   - 預覽可刪除 ignore 檔：`git clean -ndX`
   - 實際刪除 ignore 檔：`git clean -fdX`
   - 檢查整潔度：`git status --short`
