# RL_brain_trainer (V4 Mainline)

本 repo 目前主線是 **V4（sim2d）**：
以「可驗證、可重跑、可逐步升級到手臂場景」為目標，先把移動任務中的層級決策與控制約束打穩，再往 V5 manipulation 擴展。

---

## 1) 架構總覽（L1 / L2 / L3）

- **L1（語意/全域意圖）**  
  產生任務意圖與約束，不直接輸出控制量。
- **L2（局部規劃 / policy）**  
  根據狀態與目標產生局部決策（可含記憶/殘差策略）。
- **L3（deterministic follower）**  
  把 L2 指令轉成可執行控制，負責可追蹤性與穩定性。

核心契約：
> 把「理解」和「技能」分開，把「技能」和「精度」分開。

介面細節請看：`docs/V4_INTERFACE_SPEC.md`

---

## 2) V4 關鍵機制（本次更新）

### 控制約束（Control Constraints）
- 線速度：`v ∈ [-1.2, 1.2]`（允許倒車）
- 角速度：`ω ∈ [-2π, 2π]`（±360 deg/s）

設計意圖：
- 保留反向修正能力（避免只能前進導致局部幾何卡死）
- 維持轉向方向連續性，同時避免無界角速造成訓練/執行不穩

### Reward（目前實作）
由以下項目構成（詳見 `sim2d/env.py` 實作）：
- 距離/進度相關項
- 控制 effort 懲罰
- 成功獎勵（success bonus）
- 碰撞懲罰（collision penalty）

備註：目前 online_v3/v4 路徑沒有獨立命名為 backtracking loss 的顯式項，主要為 actor/critic 目標與其組合。

---

## 3) 實驗方法（V4）

### 訓練設定（代表性）
- no-obstacle MVP：`hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- complex MVP：`hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- complex + velocity constraints：`hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

### 執行方式
在 `hrl_ws` 內啟動（依對應 YAML）：

```bash
cd hrl_ws
uv sync
source .venv/bin/activate

python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

### 評估與可視化
- 以 success/collision/timeout 為主
- 保留 checkpoint 與 rollout（GIF/影片）作為行為驗證
- 建議同步保存 JSON metrics + progress log 便於對照

---

## 4) 最新結果（V4）

### No-obstacle MVP
- success rate：**95.83%**（115/120）

### Complex + Velocity Constraints（rerun）
- success rate：**95.56%**（172/180）
- collisions：**8**
- timeout：**0**

結論（目前）：
- 在開放倒車 + 有界角速度條件下，complex 場景仍維持高成功率。
- 目前主要優化方向轉向：
  1) 大 heading error 時的速度調度（更明確「先轉再走」）
  2) progress/backtracking 相關 shaping
  3) V5 手臂場景的狀態/動作介面遷移

---

## 5) 參考文獻（與實作對齊）

完整索引：`docs/literature/INDEX.md`  
建議閱讀路徑：`docs/literature/PIPELINE_GUIDE.md`

目前已歸檔且直接相關：
- **RoBridge**: Hierarchical cognition-execution 架構參考
- **Mem-α**: 記憶建構 policy 學習
- **Memory-R1**: 記憶管理 + RL 對齊
- **RLBenchNet**: 任務對應架構選型基準
- **LAC (Lyapunov Actor-Critic)**: 約束/穩定性導向 RL 設計

---

## 6) 目錄與重點檔案

### 主要程式
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/env.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`

### 主要設定
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

### 文件
- `docs/V4_INTERFACE_SPEC.md`
- `docs/V3_VELOCITY_PATCH_REPORT.md`
- `docs/V3_GUIDE.md`（V3 歷史主線）
- `docs/V3_PROGRESS_BOARD.md`

---

## 7) Branch 策略

- `main`：目前對齊 **V4 主線**
- `v3-online-memory`：保留 V3 階段成果
- `v2`：歷史基線

下一階段：**V5 模擬手臂場景（manipulation）**。
