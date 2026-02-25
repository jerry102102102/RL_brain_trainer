# V3 Velocity-Control Patch Report (2026-02-19)

## 變更目標
1) 控制介面由加速度導向改為速度命令導向（L3 輸出 `v_cmd, omega_cmd`）。
2) 場景採樣更穩定：拉開 start-goal 距離、避免障礙過度聚集/堵死主走廊。
3) 重新訓練並重跑實驗與可視化。

## Patch 內容
- `env.py`
  - 新增 `control_mode`（預設 `velocity`）
  - velocity mode 下改為一階速度追蹤動力學（含 command noise / gust / delay）
  - 新增場景穩定參數：
    - `min_start_goal_dist`
    - `min_obstacle_spacing`
    - `corridor_clearance`
- `train_rl_brainer_v3_online.py`
  - L3 follower `_rbf_controller` 改為輸出速度命令（不是 accel）
  - 由 config 注入 env 參數（control mode / 場景參數）
- `visualize_v3_episode.py`
  - 支援新 env 參數
  - 保持五角形機器人渲染
- 新增 config：
  - `train_rl_brainer_v3_online_velocity_patch_quick.yaml`
  - `train_rl_brainer_v3_online_velocity_patch_tuned.yaml`

## 訓練與實驗結果

### A) quick config（`...velocity_patch_quick.yaml`）
- train_episodes: 80
- eval_episodes: 30
- success_rate: **0.00**
- done_reasons: collision 29 / timeout 1
- tracking_rmse: 1.387
- control_effort: 41.845

### B) tuned config（`...velocity_patch_tuned.yaml`）
- train_episodes: 160
- eval_episodes: 30
- success_rate: **0.00**
- done_reasons: collision 26 / timeout 4
- tracking_rmse: 1.368
- control_effort: 57.383

> 結論：目前「訓練版 online_v3」尚未在此 patch 設定下收斂到可用 success。

## 行為可行性快速對照（非訓練權重，rollout 檢查）
使用可視化 rollout（medium / obstacle=4 / 60 episodes）得到：
- L2 模式 success_rate: **0.667**, collision_rate: **0.333**
- L3-only 模式 success_rate: **0.483**, collision_rate: **0.517**

> 解讀：在 patch 後的幾何+控制設定中，L2 仍明顯優於 L3-only；但訓練路徑仍需再調教才能通過目標挑戰。

## 產出動畫
- `/tmp/pentagon_l2_velocity_patch_success.gif`
- `/tmp/pentagon_l2_velocity_patch_failure.gif`

## 下一步建議
1) 對 online_v3 訓練增加 curriculum（先 obstacle=0/1，再升到4/8）。
2) actor target 從純 oracle MSE 改為「可行性加權」(collision-sensitive loss)。
3) 先把 eval 目標定為：Level4 成功率穩定 > baseline，再推 Level5。
