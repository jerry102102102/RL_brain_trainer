# WP0 Gap Analysis (2026-03-03)

## Scope
- Source scene project has been vendored into this repo at:
  - `external/ENPM662_Group4_FinalProject`
- WP0 smoke baseline now uses local vendored scene via:
  - `scripts/wp0_scene_smoke.sh`

## 已達成
- 場景內容已搬入 `RL_brain_trainer`，且保留結構與必要內容：
  - `external/ENPM662_Group4_FinalProject/src`
  - `external/ENPM662_Group4_FinalProject/scripts`
  - `external/ENPM662_Group4_FinalProject/README.md`
  - 場景資源（如 `worlds/v5_kitchen_empty.sdf`、`meshes/*.STL`、`models/*.sdf`）
- WP0 smoke 腳本已落地於 repo 根目錄：
  - `scripts/wp0_scene_smoke.sh`
  - 流程包含：build、launch、topic readiness、control trigger（`joint_reset_node`）
- 舊路徑預設已替換為 repo 內路徑（避免依賴外部舊目錄）：
  - `scripts/v5/bridge_kitchen_scene.sh` default source -> `external/ENPM662_Group4_FinalProject`
  - `scripts/v5/launch_kitchen_scene.sh` default scene path -> `external/ENPM662_Group4_FinalProject`
  - `docs/wp0_run.md`、`docs/WP0_DOD_CHECKLIST.md` 已切換為新基準

## 未達成
- WP0 live smoke 尚未通過：
  - `scripts/wp0_scene_smoke.sh` 實跑結果為 FAIL（卡在 `/clock` topic timeout）
- 因 launch 失敗，後續 topic 檢查與控制觸發驗證無法完整完成（屬阻塞式失敗）

## 阻塞原因
- 主要為環境權限限制，非場景程式邏輯錯誤：
  - `getifaddrs: Operation not permitted`
  - FastDDS / RTPS transport 建立失敗（SHM 與 UDP socket）
  - Gazebo log path/執行期間出現 permission denied（例如 `/home/jerry/.gz/...`）
- 證據位置：
  - `artifacts/wp0/smoke_runs/20260303_152220/launch.log`
  - `artifacts/wp0/smoke_runs/20260303_152220/colcon_build.log`（build 成功）

## 下一步（按優先級）
1. 在允許 ROS2 DDS 網路/IPC 的環境重跑 `scripts/wp0_scene_smoke.sh`（可用 host shell 或放寬 sandbox 網路限制）。
2. 若環境仍受限，嘗試切換 DDS 設定（例如禁用 SHM 或調整 `RMW`/transport）以繞過受限通道，再重跑 smoke。
3. smoke 通過後，接續跑 `scripts/v5/run_wp0_healthcheck.sh --live` 產出完整 WP0 DoD 證據，補齊 `approx_sync/tray_stability/state_latency/rosbag_replay`。
