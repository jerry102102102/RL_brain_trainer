# RLBenchNet (arXiv:2505.15040) — 歸檔筆記

- PDF: `1d22097a-9f22-4bde-af51-fadbf6df84fe.pdf`
- Title: **RLBenchNet: The Right Network for the Right Reinforcement Learning Task**
- Authors: Ivan Smirnov, Shangding Gu
- Year: 2025
- Link: https://arxiv.org/abs/2505.15040
- DOI: https://doi.org/10.48550/arXiv.2505.15040
- Pages: 15

## 一句話
這篇不是新演算法，而是「同一個 PPO 框架下比較不同 backbone（MLP/LSTM/GRU/TrXL/GTrXL/Mamba/Mamba-2）在不同任務型態的適配規則」。

## 你現在最該吸收的重點（對 L2/L3 實作）
1. **架構選擇要看任務型態，不是越複雜越好**（結論段，p.9）。
2. **Mamba-2 在 memory-heavy 任務與計算效率之間很平衡**（abstract p.1；結論 p.9；效率表 p.12-13）。
3. **Transformers 只在「真的長時記憶難題」才值得付高成本**（實務指南 p.8；結論 p.9）。
4. **在 Markov/中短時依賴任務，MLP / recurrent 往往更划算**（MuJoCo+classic control結果，p.4-6）。

## 內容錨點（page anchors）
- **整體主張 + throughput/memory headline**：p.1 Abstract
  - Mamba 4.5x throughput vs LSTM；Mamba-2/TrXL/GTrXL 才能解最難 memory 任務；Mamba-2 比 TrXL 記憶體省 8x。
- **Memory-intensive 任務結果（DoorKey / Memory-S11）**：p.7
  - Memory-S11: Mamba-2 最佳且穩定，TrXL/GTrXL 次之，LSTM/GRU/MLP 幾乎學不起來。
- **Architecture-environment compatibility + practitioner guideline**：p.8
  - 給出「先 MLP，再 Mamba-2，必要才上 LSTM/GRU/Transformer」路線。
- **總結（避免複雜度迷信）**：p.9 Conclusion
  - 明確寫到 complexity 不等於 performance，建議先用高效模型。
- **可量化效率證據**：p.12-13 Table 1/2/5/6
  - SPS、訓練時間、推理延遲、GPU memory 皆有表格可直接引用。

## 對 RL_brain_trainer 的可執行映射（MVP）
### A) L2 planner backbone 選型流程（先工程再理論）
1. **Lv1-2（短時依賴）**：先用 MLP/LSTM baseline。
2. **Lv3-4（中長依賴）**：優先測 Mamba-2（同等 budget 下）。
3. **Lv5（長時 + 高約束）**：再引入 TrXL/GTrXL 做上限對照。

### B) 你要補進 eval protocol 的四個固定欄
1. **Success / collision / timeout**（任務面）
2. **SPS + wall-clock**（效率面）
3. **GPU mem allocated/reserved**（部署面）
4. **同 checkpoint 的 rollout GIF**（可解釋面，防 heuristic 冒充）

### C) 對你目前卡點（Lv5 near-collapse）的實際意義
- 這篇支持你「不要直接把 Transformer 當預設」，先以 **Mamba-2 / recurrent** 做成本更低的主線；
- 但若 Lv5 需要極長時間 credit assignment，仍要保留 TrXL/GTrXL 作 upper-bound 參考組。

## 限制與邊界
1. 主要比較的是 **PPO + 不同 backbone**，非你目前 L1/L2/L3 分層控制架構；
2. benchmark 仍偏標準環境（MiniGrid/Atari/MuJoCo），不等同你的 pentagon collision 約束幾何；
3. 超參設定雖盡量對齊參數量，但仍可能影響個別架構上限（作者也在 future work 承認）。

## 下次檢索關鍵詞（3-8 個）
- `architecture-environment compatibility`
- `Mamba-2 memory-intensive RL`
- `PPO backbone benchmark`
- `SPS training time GPU memory RL`
- `Transformer only for long-horizon memory`
- `complexity does not imply performance`

## 建議標籤
`#architecture-selection` `#PPO-backbone` `#Mamba2` `#RL-systems` `#L2-planner`
