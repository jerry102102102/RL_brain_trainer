# QUICKSTART (V4) — 10 分鐘上手

這份文件給「第一次接手」的人：只做最短步驟，先把 V4 跑起來。

## 0) 前置條件
- Python 3.11+
- 已安裝 `uv`
- 在 repo 根目錄

## 1) 建環境
```bash
cd hrl_ws
uv sync
source .venv/bin/activate
```

## 2) 檢查關鍵路徑
```bash
ls src/hrl_trainer/hrl_trainer/sim2d
ls src/hrl_trainer/config | grep v4
```

## 3) 先跑診斷（推薦）
> 目標：先確認 L0/L1 基礎路徑沒壞。

```bash
python src/hrl_trainer/hrl_trainer/sim2d/diag_l0_rbf_straight.py
python src/hrl_trainer/hrl_trainer/sim2d/diag_l1_planner_straight.py
```

## 4) 跑 V4 主實驗（快速版）
```bash
python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v4.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml
```

## 5) 跑複雜場景設定（建議）
```bash
python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v4.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

## 6) 結果整理（最小要求）
- 記錄本輪核心 metric（success / collision / timeout）
- 保存 artifact 路徑（例如 `/tmp/v4_*.json`, checkpoint, rollout gif）
- 更新對應報告或進度板（V4 only）

## 7) 常見雷點
- 不要把 L1/L2/L3 契約混掉：L1 給語意/約束，L2 規劃，L3 deterministic follower。
- 不要讓記憶模組覆蓋安全控制器；記憶是輔助，不是取代可行性。
- 評估先看成功率與碰撞/可行性，不只看 RMSE。

---

介面契約：`docs/V4_INTERFACE_SPEC.md`

## V5 M2.4 快速命令（deterministic）

### 1) 產生 trainer_loop artifact（JSON + JSONL smoke run）
```bash
cd hrl_ws/src/hrl_trainer
python3 -m hrl_trainer.v5.artifacts \
  --episode-index 0 \
  --steps 3 \
  --json-path /tmp/v5_m2_4_smoke_artifact.json \
  --jsonl-path /tmp/v5_m2_4_smoke_artifact.jsonl
```

### 2) 跑 Rule-L2 v0 baseline benchmark 並輸出 summary JSON
```bash
cd hrl_ws/src/hrl_trainer
python3 -m hrl_trainer.v5.benchmark_rule_l2_v0 \
  --episodes 8 \
  --seed 42 \
  --output /tmp/v5_rule_l2_v0_benchmark_summary.json
```

## V5 M2.5 Eval Harness 快速命令（一行）

```bash
cd hrl_ws/src/hrl_trainer && python3 -m hrl_trainer.v5.eval_harness
```

- 預設 output path: `artifacts/reports/v5/v5_eval_rule_l2_v0_seed42_ep8.json`
- Strict RL-L2（real path in eval harness, no fallback；非實機/HIL）:
```bash
cd hrl_ws/src/hrl_trainer && python3 -m hrl_trainer.v5.eval_harness --policy rl_l2 --strict-policy
```

- 預期輸出重點欄位:
  - `policy_requested`: `rl_l2`
  - `policy_executed`: `rl_l2`
  - `fallback_used`: `false`
  - `summary.stage.benchmark_schema`: `v5_rl_l2_v0_benchmark`
  - `summary.stage.benchmark_version`: `1.0`

## V5.1 Bundle-v1（RealSAC-Gated）一次性命令

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer

# tests: reward/sac/gate
pytest -q tests/v5_1/test_reward_v1.py tests/v5_1/test_sac_update_smoke.py tests/v5_1/test_eval_sac_gate.py

# smoke train
python -m hrl_trainer.v5_1.train_loop_sac --episodes 20 --seed 20260331 --artifact-root artifacts/v5_1/train/smoke

# 30ep gate
python -m hrl_trainer.v5_1.eval_sac_gate --checkpoint artifacts/v5_1/train/smoke/checkpoint_latest.pt --episodes 30 --seed 20260331 --policy-mode sac --enforce-gates

# 100ep gate (only after 30ep GO)
python -m hrl_trainer.v5_1.eval_sac_gate --checkpoint artifacts/v5_1/train/smoke/checkpoint_latest.pt --episodes 100 --seed 20260401 --policy-mode sac --enforce-gates
```

- 若 30ep 為 HOLD：
  - policy fallback：`--policy-mode rule`
  - 套 `Bundle-v1-safe-fallback` runtime 參數（較保守權重 + 更長 warmup）
  - 保留 SAC artifacts 做離線分析，不刪檔

## V5 WP2 M2-9 一鍵重跑（推薦入口）

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
./scripts/m2_9_rerun_wp2.sh
```

預期會一次完成：
- M2-7 integration tests
- M2-8 benchmark tests
- M2-8a eval harness tests
- M2-8b formal comparison tests
- 主要輸出 JSON 產生（`artifacts/reports/v5/*.json`）

### 術語定義（重要）
- **real path**：benchmark/eval harness 的真實程式路徑執行（非 placeholder/simulated rows）。
- **real robot runtime**：實體機器人/控制器/HIL 執行。
- 本 repo 目前 WP2 M2-9 closeout 屬於前者，不代表已完成後者。

更多細節：`docs/WP2_IMPLEMENTATION_NOTE.md`
