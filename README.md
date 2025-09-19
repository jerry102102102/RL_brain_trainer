概述 / Overview

這個專案的目標是用三層式（概念層→技能層→安全控制層）的強化學習方法，訓練並部署一個可在 Gazebo/ROS 2 或純模擬環境運作的控制器。
我們提供一個單檔 Python 模組（hrl_control.py），內含 HierarchicalRLController 類別，可被 import 使用或直接當腳本跑。支援 CPU/GPU（PyTorch）。

The project implements a three-layer hierarchical RL controller that can be trained in simulation and deployed in Gazebo/ROS 2 or standalone.
It ships as a single-file Python module (hrl_control.py) exposing HierarchicalRLController, importable or runnable as a script. CPU/GPU via PyTorch are supported.

功能亮點 / Key Features

概念層 (Concept Layer)：以 DQN 學習選擇離散選項/巨集動作（options/macro-actions）。

技能層 (Skill Layer)：用 DMP（Dynamic Movement Primitives） 從零實作，將選項映射為參數化軌跡。

安全控制層 (Safety Layer)：以 CBF（Control Barrier Function）+ QP 的方式，最小化修改期望控制命令，滿足關節/速度界限與安全集合。

環境介面：不依賴 gym；提供簡易 EnvProtocol，可由你把 Gazebo/ros2_control 封裝。

ROS 2（選用）：可在有安裝 ROS 2 的環境中，訂閱 /joint_states、發布 /joint_trajectory。

可擴充：可新增自訂選項、DMP 參數、CBF 障礙／關節邊界、或替換 DQN/MPC/QP 求解器。

Concept Layer: DQN over discrete options.

Skill Layer: DMP implementation from scratch for parameterized trajectories.

Safety Layer: CBF + QP safety filter to minimally modify desired commands under constraints.

Env Interface: gym-free EnvProtocol; you can wrap Gazebo/ros2_control.

ROS 2 (optional): subscribe /joint_states, publish /joint_trajectory.

Extensible: plug new options, DMP params, CBF barriers/limits, swap DQN/MPC/QP backends.

檔案 / File

hrl_control.py — 單檔模組。匯出 HierarchicalRLController 與 HRLConfig。

以腳本執行（python hrl_control.py）會跑一個簡易 1-DoF smoke test。

Single-file module exporting HierarchicalRLController and HRLConfig.
Running it as a script performs a tiny 1-DoF smoke test.

數學骨幹 / Math Backbone

DMP

Canonical: 
𝜏
𝑠
˙
=
−
𝛼
𝑠
𝑠
τ
s
˙
=−α
s
	​

s

Transformation: 
𝜏
𝑣
˙
=
𝛼
𝑧
(
𝛽
𝑧
(
𝑔
−
𝑥
)
−
𝑣
)
+
(
𝑔
−
𝑥
0
)
𝑓
(
𝑠
)
,
𝜏
𝑥
˙
=
𝑣
τ
v
˙
=α
z
	​

(β
z
	​

(g−x)−v)+(g−x
0
	​

)f(s),τ
x
˙
=v

Forcing: 
𝑓
(
𝑠
)
=
∑
𝑖
𝜓
𝑖
(
𝑠
)
𝑤
𝑖
𝑠
∑
𝑖
𝜓
𝑖
(
𝑠
)
+
𝜀
,
 
𝜓
𝑖
=
exp
⁡
(
−
ℎ
𝑖
(
𝑠
−
𝑐
𝑖
)
2
)
f(s)=
∑
i
	​

ψ
i
	​

(s)+ε
∑
i
	​

ψ
i
	​

(s)w
i
	​

s
	​

, ψ
i
	​

=exp(−h
i
	​

(s−c
i
	​

)
2
)

CBF-QP

Safety set 
𝑆
=
{
𝑥
∣
ℎ
(
𝑥
)
≥
0
}
S={x∣h(x)≥0} with constraint

∇
ℎ
(
𝑥
)
⊤
(
𝑓
(
𝑥
)
+
𝑔
(
𝑥
)
𝑢
)
+
𝛼
ℎ
(
𝑥
)
≥
0
∇h(x)
⊤
(f(x)+g(x)u)+αh(x)≥0.

QP：最小化 
1
2
∥
𝑢
−
𝑢
des
∥
𝑅
2
+
𝜆
𝛿
𝛿
2
2
1
	​

∥u−u
des
	​

∥
R
2
	​

+λ
δ
	​

δ
2
 subject to bounds + CBF。

DQN

目標：
𝑦
=
𝑟
+
𝛾
max
⁡
𝑎
′
𝑄
𝜃
−
(
𝑠
′
,
𝑎
′
)
y=r+γmax
a
′
	​

Q
θ
−
	​

(s
′
,a
′
)；target soft update；replay buffer；epsilon-greedy。

上述全部都內建在 hrl_control.py，不需額外檔案。
All formulas are implemented inside the single module.

安裝需求 / Requirements

Python 3.9+

PyTorch 2.0+（自動使用 CUDA 若可用）

NumPy, SciPy

OSQP（或自動退回 qpsolvers）

（選用）ROS 2：rclpy, sensor_msgs, trajectory_msgs

pip install torch numpy scipy osqp qpsolvers
# ROS 2 related packages depend on your distro; optional.

快速開始 / Quick Start
1) 匯入使用 / Import
from hrl_control import HierarchicalRLController, HRLConfig
import numpy as np

cfg = HRLConfig(
    n_joints=2,
    state_dim=6,
    options=["hold","dmp_small_step","dmp_medium_step","dmp_large_step"],
    dmp_n_basis=15,
    dt=0.05
)
ctrl = HierarchicalRLController(cfg)

2) 接你的環境 / Hook your Env

你需要提供一個符合 EnvProtocol 的環境包裝（reset/step/state_dim/n_joints/goal）。
Provide a thin wrapper that matches EnvProtocol.

class MyGazeboEnv:
    def __init__(self): ...
    def reset(self) -> np.ndarray: ...
    def step(self, option_id: int): ...
    @property
    def state_dim(self) -> int: ...
    @property
    def n_joints(self) -> int: ...
    @property
    def goal(self) -> np.ndarray: ...

env = MyGazeboEnv()
ctrl.train(env, total_steps=100_000)
metrics = ctrl.evaluate(env, episodes=5)
print(metrics)

3) ROS 2（選用）/ ROS 2 (Optional)

run_ros2_closed_loop(...) 會延遲匯入 ROS。若未安裝，會丟 RuntimeError。
The method lazily imports ROS; if unavailable, it raises RuntimeError.

ctrl.run_ros2_closed_loop(
  controller_ns="/joint_trajectory_controller",
  hz=10.0,
  topic_joint_states="/joint_states",
  topic_joint_traj="/joint_trajectory"
)

設定說明 / Configuration

HRLConfig 常用欄位：

n_joints, state_dim, options：基本空間與動作集合

dmp_*：DMP 參數與 basis 數

horizon_steps, dt：DMP 展開與控制週期

gamma, lr, batch_size, replay_capacity, tau：DQN 訓練

eps_*：探索率日程

cbf_alpha, qp_R_diag, max_joint_vel, use_slack：CBF-QP 安全層

device："cuda" / "cpu" / None（自動）

訓練流程 / Training Loop

select_option：DQN 以 epsilon-greedy 選離散選項。

option_to_dmp：將選項映射成 joint-wise DMP 軌跡。

safety_layer_filter：逐步以 CBF-QP 修正速度命令。

環境回傳 reward / next state，進入 replay buffer。

train_step：抽樣 batch、計算損失、反向傳播、更新 target。

You can call:

ctrl.push_transition(s, a, r, s2, done)
ctrl.train_step()

GPU/CPU

自動偵測 CUDA：torch.device("cuda" if available)。

你也可在 HRLConfig(device="cpu") 強制 CPU。

DMP 與 QP 在 CPU；DQN 前向/反向在 GPU（若可）。

安全與限制 / Safety & Limits

安全層是最後一道保護；若 QP 不可行，會退回裁剪後的 u_des。

障礙與邊界要以 barrier 函式定義並提供梯度。請根據你的機器手臂運動學/場景設計。

在真機前務必先在模擬器充分驗證，逐步放寬限幅。

常見擴充 / Typical Extensions

更多選項：新增 "dmp_reach_to_pose", "dmp_orient", "hold_damped" …

技能學習：把 DMP 參數由策略網路產生（NDP/Actor-Param）。

MPC：用短時域 MPC 取代或輔助 QP 層。

多目標融合：以權重融合多個 DMP / RMP 任務。

偵錯建議 / Troubleshooting

QP 不收斂：降低 dt、放鬆 max_joint_vel、啟用 use_slack、檢查 barrier 梯度方向。

訓練無法收斂：調整 reward、縮小動作集合、增大 replay、檢查 state_dim 與正規化。

ROS 訊息不同步：確認 joint_names 順序一致、time_from_start 單調遞增。

授權 / License

根據你的專案政策填寫（例如 MIT/Apache-2.0）。

Choose a license that fits your project (e.g., MIT/Apache-2.0).

總結 / Summary

用戶只需：

準備符合 EnvProtocol 的環境（Gazebo/ros2_control 包裝）。

設定 HRLConfig，初始化 HierarchicalRLController。

呼叫 train() 與 evaluate()；（選用）run_ros2_closed_loop() 上線。

You only need to:

Wrap your sim into EnvProtocol.

Configure HRLConfig, create HierarchicalRLController.

Call train()/evaluate(), and optionally run_ros2_closed_loop() for online control.