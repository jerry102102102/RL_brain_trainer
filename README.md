# RL_brain_trainer (V3)

本 repo 目前以 **V3 三層分工架構**為主線，目標是把機器人任務拆成可驗證的層級因果，並用系統化消融實驗確認每層的貢獻。

## V3 架構（目前主版）

- **L1（語意/全域理解）**：輸出可被規劃器消化的意圖與約束表徵（不是控制量）
- **L2（局部規劃/技能）**：根據 L1 表徵產生可執行局部計畫（含 memory/LSTM 可選）
- **L3（deterministic follower）**：固定追蹤控制，負責精度與穩定

> 核心契約：**把「理解」和「技能」分開，把「技能」和「精度」分開**。

---

## 重要實驗結論（V3）

### 1) Hierarchy Meaning Ablation（3 seeds）
文件：`docs/V3_HIERARCHY_MEANING_ABLATION.md`

- low：A 0.889, B 1.000, C 1.000
- medium：A 0.856, B 0.967, C 0.944
- high：A 0.600, B 0.811, C 0.833

結論：修正層級語義契約後，L2/L3 分工價值明顯。

### 2) Level-5 Pentagon Benchmark（5 seeds）
文件：`docs/V3_LEVEL5_PENTAGON_ABLATION.md`

- Level1~4：L2（B/C）明顯優於 A
- Level5：A/B/C 接近失效區（~0.15 success）

結論：高難度下進入飽和區，下一步應優先補強「可行性表徵 + 約束規劃骨架」，而非單純堆 memory/LSTM。

---

## 目錄（V3 相關）

> 舊版規劃/實驗草稿已整理到 `docs/archive/legacy/`，避免干擾目前主線。

### 核心訓練/實驗程式
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/run_v3_hierarchy_meaning_ablation.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/diag_l0_rbf_straight.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/diag_l1_planner_straight.py`

### 主要設定檔
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_online_quick.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_hierarchy_meaning_ablation.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_level5_pentagon_ablation.yaml`

### 實驗文件
- `docs/V3_GUIDE.md`（建議先讀）
- `docs/V3_ONLINE_MEMORY_LSTM_PLAN.md`
- `docs/V3_PROGRESS_BOARD.md`
- `docs/L2_MEMORY_ABLATION.md`
- `docs/L2_DETERMINISTIC_PLUS_MEMORY.md`
- `docs/V3_THREE_LAYER_LSTM_ABLATION.md`
- `docs/V3_COMPLEXITY_ABLATION.md`
- `docs/V3_HIERARCHY_MEANING_ABLATION.md`
- `docs/V3_LEVEL5_PENTAGON_ABLATION.md`

---

## 執行（快速）

> 以 repo 內 `hrl_ws` 為工作目錄，依照各 yaml 啟動對應實驗。

1. 先跑層級診斷（L0/L1）
2. 再跑 hierarchy meaning ablation
3. 最後跑 level5 pentagon 壓力測試

結果請同步寫入對應 `docs/*.md` 與 JSON artifact（例如 `/tmp/v3_*`）。

---

## 目前狀態

- 主線分支：`v3-online-memory`
- 本 repo 的 `main` 應對齊此 V3 主線內容
- 舊版（早期單檔概念敘述）已退場，不再作為主說明
