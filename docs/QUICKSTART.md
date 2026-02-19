# QUICKSTART (V3) — 10 分鐘上手

這份文件給「第一次接手」的人：只做最短步驟，先把 V3 跑起來。

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
ls src/hrl_trainer/config | grep v3
```

## 3) 先跑診斷（推薦）
> 目標：先確認 L0/L1 基礎路徑沒壞。

```bash
python src/hrl_trainer/hrl_trainer/sim2d/diag_l0_rbf_straight.py
python src/hrl_trainer/hrl_trainer/sim2d/diag_l1_planner_straight.py
```

## 4) 跑 V3 主實驗（快速版）
```bash
python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py \
  --config src/hrl_trainer/config/train_rl_brainer_v3_online_quick.yaml
```

## 5) 跑關鍵消融（建議順序）
```bash
# hierarchy meaning
python src/hrl_trainer/hrl_trainer/sim2d/run_v3_hierarchy_meaning_ablation.py \
  --config src/hrl_trainer/config/train_rl_brainer_v3_hierarchy_meaning_ablation.yaml

# level5 pentagon
python src/hrl_trainer/hrl_trainer/sim2d/run_v3_hierarchy_meaning_ablation.py \
  --config src/hrl_trainer/config/train_rl_brainer_v3_level5_pentagon_ablation.yaml
```

## 6) 結果整理（最小要求）
- 更新 `docs/V3_PROGRESS_BOARD.md`（TODO/DOING/DONE）
- 把本輪結果補到對應報告：
  - `docs/V3_HIERARCHY_MEANING_ABLATION.md`
  - `docs/V3_LEVEL5_PENTAGON_ABLATION.md`
- 若有 JSON 輸出，保存 artifact 路徑（例如 `/tmp/v3_*.json`）

## 7) 常見雷點
- 不要把 L1/L2/L3 契約混掉：L1 給語意/約束，L2 規劃，L3 deterministic follower。
- 不要直接把 memory/LSTM 當成覆蓋控制器。
- 看結果先看成功率與碰撞/可行性，不只看 RMSE。

---

更多背景與實驗導覽：`docs/V3_GUIDE.md`
