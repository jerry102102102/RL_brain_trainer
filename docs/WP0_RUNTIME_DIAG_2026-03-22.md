# WP0 Runtime Diagnostic — 2026-03-22

## Scope
驗證 `joint_reset_node` 在場景啟動後是否能正常發佈 reset trajectory，並記錄可重現命令與證據。

## Environment
- Repo: `/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer`
- Scene workspace: `external/ENPM662_Group4_FinalProject/src`
- ROS: Jazzy

## Commands Run
```bash
# 1) Launch scene (headless)
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch kitchen_robot_description gazebo.launch.py headless:=true

# 2) In another shell, capture one trajectory message
source /opt/ros/jazzy/setup.bash
source install/setup.bash
timeout 12 ros2 topic echo /arm_controller/joint_trajectory --once

# 3) Trigger reset publisher
source /opt/ros/jazzy/setup.bash
source install/setup.bash
timeout 20 ros2 run kitchen_robot_controller joint_reset_node \
  --ros-args \
  -p publish_repeat_count:=3 \
  -p publish_repeat_period:=0.2 \
  -p time_to_reach:=1.0
```

## Evidence (published reset trajectory)
`joint_reset_node` stdout:
```text
[INFO] [1774214191.006580240] [joint_reset_node]: Published reset trajectory (first time) to 7 joints: array('d', [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0])
[INFO] [1774214191.599633857] [joint_reset_node]: Finished publishing reset trajectory.
```

`/arm_controller/joint_trajectory` sample:
```text
joint_names:
- Rack_joint
- robot_base_joint
- shoulder1_joint
- shoulder2_joint
- wr1_joint
- wr2_joint
- wr3_joint
points:
- positions:
  - 0.0
  - 0.0
  - 1.5
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  time_from_start:
    sec: 1
    nanosec: 0
```

## Result
- `joint_reset_node`: **PASS**
- Reset trajectory publication observed on `/arm_controller/joint_trajectory`: **PASS**
