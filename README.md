# RL_brain_trainer (V5 Active)

> 目前主線是 **V5 三層架構（L1/L2/L3）** 的 kitchen manipulation。
> V4/sim2d 保留為歷史基線，不是當前開發主戰場。

---

## 中文（Current Truth）

## 1) 當前狀態（2026-03）
- ✅ **WP1.5 runtime hotfix 已落地並合併到 `v5`**
  - Jazzy 環境下以 **QUASI_DEDICATED** tray pose 路徑運作（非 Pose_V dedicated）。
  - `tray_pose_adapter` 已加 deterministic gate，實測可達 `fallback_ratio=0.000`。
- ✅ **controller auto bring-up 已修復**
  - 啟動後可自動 `LOAD -> CONFIGURE -> ACTIVATE`，不需手動 spawner。
- ✅ 健康檢查關鍵項目可通過
  - `/controller_manager/list_controllers`：`joint_state_broadcaster` + `arm_controller` active
  - `/tray1/pose` sample PASS
  - `/v5/perception/object_pose_est` sample PASS

## 2) 架構與契約（L1 / L2 / L3）
- **L1（Intent）**：輸出任務語義，不輸出控制軌跡
  - Topic: `/v5/intent_packet`
- **L2（Policy / Skill）**：輸出 SkillCommand（中階決策）
  - Topic: `/v5/skill_command`
- **L3（Deterministic Executor + Safety）**：把 SkillCommand 轉成控制器可執行命令
  - Topic: `/arm_controller/joint_trajectory`

> 硬邊界：L2 不直接輸出 trajectory chunk/spline/joint trajectory。

## 3) 目前 canonical pipeline（sim）
Gazebo / bridge / perception：

`/world/empty/pose/info`
→ `/tray_tracking/pose_stream`
→ `tray_pose_adapter_node`
→ `/tray1/pose`
→ `object_id_publisher_node`
→ `/v5/perception/object_pose_est`

> 備註：在當前 Jazzy 環境，`Pose_V` dedicated 路徑不可用，因此使用 QUASI_DEDICATED（deterministic legacy adapter）。

## 4) WP2 方向（已拍板）
- L2/L3 頻率契約：
  - **L2 = 10–20 Hz**（中階連續策略）
  - **L3 = 100–200 Hz**（deterministic + safety shield）
- 互動策略：stale timeout + interpolation + predictive clamp + fail-safe fallback
- M1：Rule-L2 v0 baseline（U-slot flow）

## 5) 常用命令
### 啟動場景（headless）
```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.bash
source external/ENPM662_Group4_FinalProject/src/install/setup.bash
ros2 launch kitchen_robot_description gazebo.launch.py headless:=true use_software_renderer:=true
```

### 快速健康檢查
```bash
source /opt/ros/jazzy/setup.bash
source /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/external/ENPM662_Group4_FinalProject/src/install/setup.bash

ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers "{}"
ros2 topic echo /tray1/pose --qos-reliability best_effort --once
ros2 topic echo /v5/perception/object_pose_est --qos-reliability best_effort --once
```

---

## English (Brief)

### Current active line
- V5 kitchen manipulation is the active development line.
- WP1.5 runtime fixes are merged on `v5`.
- In current ROS 2 Jazzy environment, Pose_V dedicated tray path is unavailable; system runs in **QUASI_DEDICATED** mode with deterministic legacy adapter and validated healthy topic flow.

### Core contract
- L1 intent: `/v5/intent_packet`
- L2 policy skill command: `/v5/skill_command`
- L3 deterministic execution: `/arm_controller/joint_trajectory`

### Health checks expected
- `joint_state_broadcaster` + `arm_controller` active
- `/tray1/pose` sample available
- `/v5/perception/object_pose_est` sample available (`tray1`)

For detailed implementation milestones and constraints, see:
- `docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md`
- `docs/WP1_5_RUNTIME_PARITY_CHECKER.md`
- `docs/V5_EXPERIMENT_LOG.md`
