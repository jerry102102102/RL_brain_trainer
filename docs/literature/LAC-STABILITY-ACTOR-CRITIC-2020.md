# LAC Stability Actor-Critic (2020) — 歸檔筆記

- PDF: `78f52902-f3d2-4a81-a0e1-0eec28e7fb26.pdf`
- Title: **Actor-Critic Reinforcement Learning for Control with Stability Guarantee**
- Authors: Minghao Han, Lixian Zhang, Jun Wang, Wei Pan
- Year: 2020
- Venue/Type: conference-style paper (12 pages)

## 一句話
把 Lyapunov 穩定條件直接嵌進 actor-critic（LAC），讓 policy 學習時就同時滿足「控制表現 + 閉環穩定」的約束。

## 你現在主線最可用的重點
1. **不是只追 reward**：他們把「energy decreasing condition」做成可 sample 驗證的 critic 條件（對你的 safety/collision 指標很有參考性）。
2. **穩定條件可資料化**：不需要完整動態模型，也能用資料分佈近似檢查 Lyapunov 條件。
3. **robustness 是結果不是口號**：文中在參數擾動與外部干擾下，LAC 比一般 SAC 更能回到目標附近。

## 內容錨點（page anchors）
- **問題動機（RL 無穩定保證）**：p.1 Introduction
- **穩定定義（mean cost stability）**：p.2 Definition 1
- **核心理論條件（Lyapunov inequalities）**：p.3 Eq. (1)(2)
- **Lyapunov critic 與 target 設計**：p.4 Eq. (8)(9)
- **LAC constrained optimization 形式**：p.4-5 Eq. (10)(11)(12)
- **實驗總覽（CartPole/MuJoCo/GRN + baseline 比較）**：p.5-6
- **擾動/參數不確定性 robustness 結果**：p.6

## 對 RL_brain_trainer 的可執行映射（MVP）
### A) L2 層加入「穩定 critic」
- 在你現有 task reward 外，增加 Lyapunov-style 安全項：
  - `ΔL = E[L(s_{t+1}) - L(s_t)]`
  - 訓練目標偏向 `ΔL + α * safety_cost <= 0`（sample 平均下）

### B) 用在你 Lv4/Lv5 卡點的具體改法
1. 把 collision / near-collision 當 `safety_cost`。
2. 設計一個可學 Lyapunov proxy（例如局部幾何風險場 + 速度誤差組合）。
3. 在 L2 planner policy loss 中加 constraint penalty（類 Lagrangian 形式）。
4. 評估時除 success rate，再固定輸出 `E[ΔL]` 與 violation ratio。

### C) 與你 L1/L2/L3 結構的對齊
- L1：仍負責語意與高階約束
- L2：引入「穩定約束下的局部規劃 policy」
- L3：維持 deterministic follower（不變）

## 限制與邊界
1. 這篇以經典 control/模擬任務為主，不是你的三層 HRL 直接解法。
2. Lyapunov 函數（或 proxy）設計品質，會直接決定方法上限。
3. 理論條件在 sample 近似下成立，仍需你用高難度幾何關卡做壓力測試。

## 下次檢索關鍵詞（3-8 個）
- `Lyapunov actor-critic LAC`
- `energy decreasing condition RL`
- `stability guarantee model-free control`
- `robustness under disturbances RL control`
- `mean cost stability MDP`
- `safe actor-critic Lyapunov constraint`

## 建議標籤
`#safe-rl` `#lyapunov` `#actor-critic` `#L2-safety` `#robust-control`
