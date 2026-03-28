# V5.1 REAL SAC 任務票（從骨架切到真訓練）

> 目的：把目前 `pipeline_e2e` 的 deterministic/pseudo metrics 替換成真正 SAC 訓練與真實 reward 評估。

## R1 — Reward v1 落地（真實計算，不再 pseudo）
- Goal: 實作 SAC 真實 reward function，替代目前 `success_rate = 0.5 + 0.1*ep`。
- Scope:
  - 在 `v5_1/` 建立 reward 計算模組（例如 `reward.py`）。
  - 每 step 輸出 reward components 與總分。
- Reward v1:
  - `+ progress_delta`
  - `- action_norm_penalty`
  - `- jerk_penalty`
  - `- intervention_penalty`
  - `- clamp_projection_penalty`
  - `- timeout_or_reset_fail_penalty`
  - `+ episode_success_bonus`
- DoD:
  - `pipeline_e2e` 改用真實 reward，移除 pseudo success/intervention 生成。
  - `pipeline_summary.json` 含 reward 統計（mean/std/min/max）。
- Validation:
  - 單測：reward 組件與符號方向正確。
  - 集成：10 episodes 產生非固定/可變 reward 序列。
- Artifacts:
  - `artifacts/v5_1/e2e/<run_id>/reward_trace.jsonl`
- Owner: je
- Priority: P0
- Dependencies: T1~T8 completed
- Rollback: 可用 `--reward-mode pseudo` 臨時回退（只作 debug）

## R2 — SAC 訓練核心（Actor/Twin-Q/Replay）
- Goal: 在 `v5_1` 上線最小可訓練 SAC。
- Scope:
  - `sac_torch.py`（actor + twin critics + alpha，torch）
  - `replay_buffer.py`
  - `trainer.py`（sample/update/target soft-update）
- DoD:
  - 可執行 1 個完整 train run（收集->更新->輸出 checkpoint）。
  - 輸出 loss 指標（actor_loss/critic_loss/alpha）。
- Validation:
  - 單測：tensor shape、update step 不報錯。
  - Smoke：最小 run 可產生 checkpoint。
- Artifacts:
  - `artifacts/v5_1/train/<run_id>/checkpoints/`
  - `artifacts/v5_1/train/<run_id>/train_metrics.jsonl`
- Owner: je
- Priority: P0
- Dependencies: R1
- Rollback: `--mode eval_only` 關閉訓練更新

## R3 — 真實 L2 Policy 接線（torch 單一路線）
- Goal: L2 使用 SAC Torch policy action。
- Scope:
  - `pipeline_e2e` 串接 SAC Torch actor 輸出 `delta_q`。
  - `pipeline_e2e` 固定 `--policy-mode sac_torch`。
- DoD:
  - `--policy-mode sac_torch` 可運行並產出 action log。
- Validation:
  - 測試：torch 路線（pipeline/sac_torch/reward/gates）可跑。
- Artifacts:
  - `artifacts/v5_1/e2e/<run_id>/policy_trace.jsonl`
- Owner: je
- Priority: P0
- Dependencies: R2
- Rollback: 不提供 numpy/rule fallback；僅允許回退到前一個 torch checkpoint

## R4 — 真實成功判定與終止條件
- Goal: 用真場景指標定義 success/fail/timeout，而非假資料。
- Scope:
  - success：goal 位置誤差 + 安全條件
  - fail：reset fail / watchdog hard stop / timeout
- DoD:
  - `pipeline_summary.json` 的 success/intervention 皆來自真判定。
- Validation:
  - 測試：三種終止路徑（success/fail/timeout）。
- Artifacts:
  - `artifacts/v5_1/e2e/<run_id>/episode_outcomes.jsonl`
- Owner: je
- Priority: P0
- Dependencies: R1, R3
- Rollback: debug 時允許 `--success-mode legacy`（預設關）

## R5 — Gate 指標改真實資料源
- Goal: T8 gates 的 metrics 來源改為真訓練/真執行資料。
- Scope:
  - P0/P1 指標由真實 episode outcomes + logs 計算。
- DoD:
  - `gate_result.json` 不再依賴 pseudo series。
- Validation:
  - 測試 pass/fail 案例使用真實 metrics fixture。
- Artifacts:
  - `artifacts/v5_1/e2e/<run_id>/gate_result.json`
- Owner: je
- Priority: P0
- Dependencies: R4
- Rollback: `--gate-source legacy` 僅用於診斷

## R6 — 10/30 Episodes 真驗收批
- Goal: 以真環境執行 10 與 30 episodes，確認穩定與趨勢。
- Scope:
  - run A: 10 episodes（快速驗收）
  - run B: 30 episodes（穩定性）
- DoD:
  - 產出兩組 artifacts + gate 判定 + 層級 logs。
- Validation:
  - reset fail-fast=0
  - logs 完整
  - success trend 非惡化
- Artifacts:
  - `artifacts/v5_1/e2e/main_v5_1_gate10_real/`
  - `artifacts/v5_1/e2e/main_v5_1_gate30_real/`
- Owner: main
- Priority: P1
- Dependencies: R1~R5
- Rollback: 若 fail，回到 R1~R5 debug

## R7 — Reward/Gate 可視化摘要
- Goal: 讓你一眼看懂 reward 與 gate 為何過/不過。
- Scope:
  - `log_summary` 擴充 reward/gate 專欄。
- DoD:
  - 產出 `run_report.md`（核心 metric + 異常點）。
- Validation:
  - 兩組 run 均可生成摘要。
- Artifacts:
  - `artifacts/v5_1/reports/<run_id>_report.md`
- Owner: je
- Priority: P1
- Dependencies: R6
- Rollback: 保留原 summary

## R8 — 文檔與操作手冊收斂
- Goal: 更新 README / V5.1 文件，明確「已進入真 SAC 版」。
- Scope:
  - 文件更新與命令範本同步。
- DoD:
  - README 有一鍵 train/eval 指令。
- Validation:
  - 新人可按文檔重跑。
- Artifacts:
  - `reports/V5_1_REAL_SAC_DELIVERY.md`
- Owner: je
- Priority: P2
- Dependencies: R1~R7
- Rollback: N/A
