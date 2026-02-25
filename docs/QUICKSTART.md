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
