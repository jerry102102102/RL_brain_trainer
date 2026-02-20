# Mem-α (arXiv:2509.25911) — 歸檔筆記

- PDF: `a4a27341-4a4b-4b09-867a-c555ea5021e6.pdf`
- Title: **Mem-α: Learning Memory Construction via Reinforcement Learning**
- Authors: Yu Wang et al.
- Year: 2025
- Link: https://arxiv.org/abs/2509.25911
- DOI: https://doi.org/10.48550/arXiv.2509.25911

## 一句話
用 RL 直接優化「記憶寫入策略」(寫什麼、怎麼寫、何時更新)，而不是只靠 prompt + tool instruction。

## 核心方法（你現在最值得借鏡的點）
1. 把 memory construction 視為 sequential decision problem：
   - 每個 chunk 都要做一組 memory action（insert / update / delete）。
2. Reward 設計是多目標：
   - `r1`：最終 QA 正確率（主目標）
   - `r2`：tool-call 格式是否正確（可執行性）
   - `r3`：compression（記憶長度懲罰）
   - `r4`：memory content quality（語意有效性）
3. 用 GRPO 訓練 policy，且移除 KL 讓策略更敢探索。
4. 記憶架構分成 Core / Semantic / Episodic（三種槽位 + 各自操作規則）。

## 主要結果（論文聲稱）
- 在多資料集上優於 Long-context / RAG baseline / 既有 memory-agent。
- 訓練長度 30k tokens，測試可泛化到 400k+（13x）。
- 對小模型（Qwen3-4B）提升明顯，接近或超過部分較大閉源 baseline 設定。

## 對 RL_brain_trainer 的可遷移價值（重點）
> 這篇不是機器人控制 paper，但「L2 記憶何時介入」的訓練哲學很能借。

### 可直接搬的設計模式
1. **把 L2 memory 介入當成 action policy**，不要當固定 heuristic。
2. **定義多 reward**，避免只看 success rate：
   - 任務達成（主目標）
   - 規劃合法性（不撞 / 不違反約束）
   - 記憶使用成本（避免過度依賴 memory）
   - 記憶內容有效性（引用到的記憶是否真的改善當下局部規劃）
3. **保留 deterministic backbone**，但把「何時讀記憶、讀哪段、如何融合」交給可學習策略。

### 對你 L1/L2/L3 的對齊建議
- L1：產生語意任務包（目標、約束、偏好）
- L2：學習「局部規劃 + 記憶調度」策略（可參考 Mem-α reward decomposition）
- L3：維持 deterministic follower，不讓 learning 直接污染低層穩定器

## 本篇限制（你要先防呆）
1. 評估主軸是 QA，不是閉環控制穩定性。
2. `r4` 用外部大模型判定 memory quality，機器人場景若照搬成本高。
3. memory 壓縮 reward (`r3`) 權重過高會犧牲任務成功（論文 ablation 也有跡象）。

## 建議放進你下一版 L2-RL spec 的三條
1. 把「memory on/off + memory source selection」明確列為 L2 action space。
2. Reward 至少拆四項：`task`, `safety`, `memory_cost`, `memory_effectiveness`。
3. 報告強制附上：
   - memory 觸發率
   - 每回合 memory token 成本
   - 有/無 memory 介入下的 collision 與 timeout 差異

---

## 建議標籤
`#memory-policy-learning` `#RL` `#L2-design` `#reward-shaping` `#generalization`
