# HRL Workspace Guide / HRL 工作區指南

本工作區 (`hrl_ws`) 匯集了操控模擬、ROS 2 橋接與分層式強化學習訓練等元件，讓您可以在 Gazebo 模擬器中訓練並驗證階層式控制策略。以下內容提供資料夾導覽、模組職責、關鍵實作概念與操作步驟。

## 目錄導覽與整體架構

```
hrl_ws/
├── README.md              ← 本說明
├── pyproject.toml         ← 使用 uv 建立 Python 環境的相依設定
├── src/
│   ├── hrl_control_bridge/ ← ROS 2 節點，橋接 Gazebo 與 HRL 訓練流程
│   ├── hrl_gazebo/         ← 模擬資產、URDF 與啟動檔，負責啟動 Gazebo 場景
│   └── hrl_trainer/        ← 強化學習控制器、訓練與推論節點
└── install/, build/, log/  ← (建置後產生) colcon build 產物
```

### 資料夾職責總覽

| 目錄 | 主要內容 | 說明 |
|------|----------|------|
| `hrl_control_bridge/config/bridge.yaml` | 桥接節點的 ROS 參數 | 控制頻率、關節限制、獎勵設計等皆在此定義。
| `hrl_control_bridge/nodes/hrl_control_bridge.py` | 橋接節點實作 | 將關節狀態轉換為強化學習觀測，並發布控制命令與獎勵。
| `hrl_gazebo/launch/sim_bringup.launch.py` | Gazebo 啟動檔 | 生成機器人、載入 `ros2_control`，並啟動控制器。
| `hrl_gazebo/urdf/` | 機械手 URDF 與控制設定 | `manipulator.urdf.xacro` 描述機械手，`ros2_control.yaml` 定義控制器與硬體介面。
| `hrl_trainer/hrl_brain_trainer/` | HRL 核心演算法 | 包含階層式 DQN、DMP 與 CBF-QP 安全層實作。
| `hrl_trainer/config/train.yaml` | 訓練參數 | 封裝節點參數，例如 option 集合、回放緩衝大小、TensorBoard 設定等。
| `hrl_trainer/nodes/hrl_trainer.py` | 訓練 ROS 節點 | 管理資料流程、回報統計、儲存檢查點。
| `hrl_trainer/nodes/hrl_policy_node.py` | 推論 ROS 節點 | 在只推論模式下載入已訓練策略並產生指令。
| `hrl_trainer/checkpoints/`、`logs/` | 輸出產物 | 訓練過程中儲存模型與 TensorBoard 記錄。

## 模組實作亮點

### hrl_control_bridge
* 將 `/joint_states` 轉換為關節角、速度與目標資訊的觀測向量 (`[q_norm, dq_clip, goal - q, goal]`)。
* 提供重設服務 `/hrl/reset`，並在重設時隨機取樣初始姿態以增加探索多樣性。
* 依 `command_interface` 自動裁剪位置或速度命令，並維護成功視窗判斷收斂。
* 透過參數設定懲罰項 (時間、slack、碰撞) 與成功獎勵，使獎勵形狀化更加直觀。

### hrl_gazebo
* `sim_bringup.launch.py` 同時啟動 Gazebo、`robot_state_publisher`、`ros2_control_node` 與控制器。
* URDF (`manipulator.urdf.xacro`) 與 `ros2_control.yaml` 定義了關節屬性與對應的 `forward_position_controller`，確保可與橋接節點對接。
* `worlds/empty.world` 為預設場景，可依需求替換或擴充障礙物與感測器。

### hrl_trainer
* `HRLTrainerNode` 將 ROS 觀測整合成 PyTorch 張量，驅動 `HierarchicalRLController`。
* `train.yaml` 中的 option 參數 (`option_fracs`, `option_taus`, `option_horizons`) 會被轉換為 `OptionSpec`，對應不同時間尺度與目標偏移的 DMP。
* 訓練節點支援 warmup、epsilon-greedy 探索、定期儲存檢查點與 TensorBoard 紀錄。
* `HRLPolicyNode` 繼承訓練節點但強制 `training_mode=eval`，並偏好載入 `final.pt` 進行純推論。

## 建置與環境準備

### 1. 透過 uv 建立 Python 相依套件
1. 安裝 [uv](https://github.com/astral-sh/uv) 後切換至工作區：
   ```bash
   cd hrl_ws
   uv sync
   ```
2. `uv` 會在 `.venv/` 內建立虛擬環境並安裝 `pyproject.toml` 中列出的套件 (`numpy`, `torch`, `scipy`, `tensorboard`, `qpsolvers`, `osqp` 等)。
3. 啟用環境 (選用)：
   ```bash
   source .venv/bin/activate
   ```
> **注意：** `rclpy`、`gazebo_ros`、`ros2_control` 等 ROS 2 套件仍需透過系統套件管理員或 ROS 2 發行版安裝。

### 2. 編譯 ROS 2 工作區
```bash
colcon build --symlink-install
source install/setup.bash
```
建議在每個新終端皆 `source install/setup.bash` 以匯入自訂套件。

## 執行流程 (How to Run)

1. **啟動 Gazebo 模擬**
   ```bash
   ros2 launch hrl_gazebo sim_bringup.launch.py
   ```
   可透過 `use_gui:=false` 無頭執行，或替換 `world:=<path>` 自訂場景。

2. **啟動控制橋節點**
   ```bash
   ros2 run hrl_control_bridge hrl_control_bridge --ros-args \
     -p joint_names:="['joint1','joint2','joint3','joint4']" \
     --params-file src/hrl_control_bridge/config/bridge.yaml
   ```
   節點會負責重設服務、發布觀測與獎勵、接收高階指令並轉成控制命令。

3. **啟動 HRL 訓練節點**
   ```bash
   ros2 run hrl_trainer hrl_trainer --ros-args \
     --params-file src/hrl_trainer/config/train.yaml
   ```
   * 訓練進度會輸出至 `logs/` (TensorBoard) 與 `checkpoints/`。
   * 觀測到的 option 選擇會以 `std_msgs/String` 發佈到 `/hrl/option_debug`。

4. **純推論模式 (選用)**
   ```bash
   ros2 run hrl_trainer hrl_policy_node --ros-args \
     --params-file src/hrl_trainer/config/train.yaml
   ```
   選擇在 `checkpoints/final.pt` 或 `latest.pt` 存在時載入並執行策略，不會進行訓練或寫入 TensorBoard。

## 模擬與測試建議

* **重設流程**：訓練節點呼叫 `/hrl/reset` 後，橋接節點會重設 Gazebo 世界並發佈初始命令；確保 `reset_world` 或 `gazebo/reset_world` 服務已註冊。
* **獎勵調校**：使用 `bridge.yaml` 中的 `progress_reward_scale`、`time_penalty`、`slack_penalty` 等參數調整學習行為。
* **目標設定**：訓練節點會根據 option 自行產生目標；若要手動指定，可向 `/hrl/goal` 發佈 `Float64MultiArray`。
* **安全監控**：`publish_slack=true` 時會發布 slack 值，可在 RViz 或 rqt_plot 監看，以判斷 QP 是否常需放鬆約束。

## 自訂與擴充

* **加入新 option**：修改 `train.yaml` 的 `option_*` 參數或在程式內自訂 `OptionConfig`，以加入新的時間尺度或目標偏移設定。
* **替換機構**：更新 `hrl_gazebo/urdf/` 下的模型與 `ros2_control.yaml`，並同步更新 `bridge.yaml` 的關節名稱與限制。
* **切換命令介面**：將 `command_interface` 設為 `velocity` 時，橋接節點會改用速度限制裁剪輸出，並於重設時送出零速度命令。
* **整合實機**：保持話題命名與 ROS 2 介面一致即可，必要時可在 `hrl_control_bridge` 增加訂閱或發佈欄位。

---
如需更深入的演算法說明與理論背景，請參考專案根目錄的 `README.md`。祝操作順利！
