# RL Brain Trainer 三層架構 v2（研究模式草案）

> 目標：把你提出的「高層（大模型決策）—中層（記憶/時序）—底層（控制）」變成**可落地、可驗證、可迭代**的系統設計。  
> 狀態：設計版（Design Spec），尚需實驗驗證與消融確認。

---

## 0) 問題定義與成功條件

### 研究目標
在開放場景下，讓機器人能完成複合操作任務（例如 reach/grasp/place/open/slide），同時兼顧：
1. 高層語義決策能力（任務理解與分解）
2. 中層時序適應能力（記憶、策略切換、抗干擾）
3. 低層控制穩定性（追蹤、收斂、魯棒）

### 成功條件（v2）
- 對比單層 baseline，三層架構在至少 1 個 scenario 上達成：
  - 成功率上升
  - 收斂時間下降
  - 擾動恢復時間下降
- 具備可重現實驗設定（配置、指標、日誌、版本）。

---

## 1) 架構總覽（你的方向落地）

## Layer-1 高層 Strategic（預設大模型決策）
**角色**：做「語義理解 + 任務分解 + 子目標規劃」，不直接輸出馬達控制。  
**輸入**：任務指令、觀測摘要（可含視覺語義 token）。  
**輸出**：
- task_id / option_id
- 子目標（subgoal）
- 終止條件（termination condition）
- 安全/邏輯約束（constraint hints）

**關鍵原則**：
- 高層預設為 pre-trained/frozen（v2 不做重訓）
- 輸出應為「可執行語義」而非低層 torque

> 參考脈絡：RoBridge（高層認知 + symbolic bridge + embodied agent）指出高層與低層解耦可提升泛化與可控性。

---

## Layer-2 中層 Tactical（主要訓練層）
**角色**：把高層語義轉成「可跟隨的期望軌跡/技能參數」，並用記憶提升連續決策品質。  
**輸入**：
- option_id / subgoal
- 歷史觀測序列 \(x_{t-n:t}\)
- 過去行動與結果（可選）

**候選模型**：Memory-based + LSTM/GRU（主）
- 記憶檢索：找相似任務片段
- 初始化：用歷史經驗加速 policy 冷啟動
- 時序編織：輸出平滑的 \(q_{desired}\) 或 skill latent

**輸出**：
- 期望軌跡 \(q_{desired}\)
- 或低維 skill code + horizon

> 參考脈絡：Memory-based learning 在 state-action constraints 任務中常比純 model-free / model-based RL 更快收斂。

---

## Layer-3 底層 Execution（控制與安全層）
**角色**：追蹤中層輸出，並在不確定動力學/擾動下保持穩定。  
**輸入**：\(q_{desired}\)、當前狀態 \((q,\dot q)\)、誤差 \(e\)。  
**輸出**：控制命令 \(u\) 或 \(\tau\)。

**控制器建議（v2）**
1. **主線**：PID/LQR（穩定基線）
2. **增強線**：RBF/LWR 局部補償（降低非線性誤差）
3. **進階線**：MRAC + TDE + NN（MRAC-NNTDE）
   - TDE 估未知動力學
   - NN 補償 TDE 殘差
   - Lyapunov 證穩（理論依據）

> 參考脈絡：IJOCTA 2025 的 MRAC-NNTDE 對不確定機械手有可行性與穩定性分析。

---

## 2) 層間介面（避免耦合失控）

### Interface A：L1 -> L2（語義介面）
```yaml
option_id: str
subgoal: {type: pose|waypoint|event, value: ...}
termination: {metric: rmse|event, threshold: ...}
constraints: ["avoid_collision", "keep_gripper_closed", ...]
```

### Interface B：L2 -> L3（控制介面）
```yaml
q_desired: [ ... ]
horizon: int
stiffness_hint: float   # optional
safety_margin: float    # optional
```

### Interface C：L3 -> L2/L1（回饋介面）
```yaml
tracking_rmse: float
recovery_time: float
constraint_violation: bool
status: normal|degraded|fail
```

---

## 3) 訓練與執行策略（避免一次做太重）

## v2 先做（推薦）
1. 凍結 L1（先不微調）
2. 主要訓練 L2（memory + LSTM）
3. L3 先用穩定基線（PID/LQR + 可選 RBF）
4. 再逐步加 MRAC-NNTDE 作增強

## 為什麼這樣做
- 降低調參維度，先確保系統打得通
- 讓改進來源可被歸因（知道是哪層帶來提升）

---

## 4) 實驗設計（可驗證）

## Scenario（先從小場景）
- 固定目標 reach/track + 中途擾動（force pulse / sensor noise / friction change）

## 指標（至少這 5 個）
1. Success Rate
2. Time-to-Convergence
3. Tracking RMSE
4. Disturbance Recovery Time
5. Control Effort \(\int ||u|| dt\)

## Ablation（最小必要）
- A0：Baseline（無 memory、無補償）
- A1：+ Tactical memory（L2）
- A2：+ RBF/LWR compensation（L3）
- A3：+ MRAC-NNTDE（L3進階）

## 驗收
- A1 對 A0：收斂或成功率至少一項顯著改善
- A2/A3：在擾動下 recovery 更快或 effort 更低

---

## 5) 與現有 repo 對齊（RL_brain_trainer）

對齊你目前 `hrl_ws` 結構：
- `hrl_trainer/`：承載 L2（policy + memory + training loop）
- `hrl_control_bridge/`：承接 L2-L3 介面，負責狀態/命令轉換
- `hrl_gazebo/`：scenario 與擾動注入

建議新增模組（後續實作）：
- `hrl_trainer/memory/episodic_memory.py`
- `hrl_trainer/policies/tactical_lstm.py`
- `hrl_trainer/controllers/execution_stack.py`（PID/RBF/MRAC 切換）
- `hrl_trainer/eval/ablation_runner.py`

---

## 6) 潛在風險與改進方向

1. **L1 計畫與物理可行性不一致**  
   - 緩解：加入可行性檢查器（pre-execution validator）
2. **L2 記憶污染（錯誤經驗反覆引用）**  
   - 緩解：memory scoring + 新鮮度衰減 + 失敗樣本標記
3. **L3 高階自適應控制調參成本高**  
   - 緩解：先基線控制器，最後再加 MRAC-NNTDE
4. **歸因困難（不知道哪層有效）**  
   - 緩解：嚴格 ablation + 固定 random seed + 統一 logging

---

## 7) v2 下一步（可執行）

### 這週最小交付
1. 完成三層介面定義（A/B/C）
2. 跑 A0、A1 兩組對比
3. 輸出第一版圖表（成功率、收斂時間）

### 30 分鐘啟動動作（今天就能做）
- 在 repo 建立 `config/exp_v2_a0.yaml` 與 `config/exp_v2_a1.yaml`
- 確認 logger 可同時記錄 RMSE / recovery / effort

---

## 8) 結論
這個 v2 三層架構是**可行**的：
- 高層保留大模型的 declarative 優勢
- 中層承擔主要學習與泛化
- 底層用控制理論保障穩定與安全

它不保證一次到位，但已具備「可落地 + 可驗證 + 可迭代」的工程形態，適合直接作為你 RL final project 的核心設計稿。
