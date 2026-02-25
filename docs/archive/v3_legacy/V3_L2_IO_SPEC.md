# V3 L2 I/O Spec (for VLM-L1 + Local Planner-L2)

## 目標
把 L2 明確定義為「局部路徑規劃器」，避免與 L1 重疊。

---

## L1 -> L2 輸入結構（建議）

```yaml
l1_packet:
  task_intent: string                 # 例如 reach_goal / avoid_dense_zone
  global_goal_xy: [float, float]
  corridor_polyline: [[x,y], ...]     # 稀疏全局走廊
  risk_map_summary:
    high_risk_zones: [[x,y,r], ...]
    preferred_side: left|right|none
  constraints:
    min_clearance: float
    max_curvature: float
    speed_hint: float
  confidence: float                   # L1 對自身輸出的信心
  timestamp: int
```

## 環境觀測輸入（L2 local context）

```yaml
l2_obs:
  ego_state: [x, y, yaw, v, omega]
  nearest_obstacles: [[x,y,r], ...]   # K 個
  local_occupancy_patch: tensor|grid  # 可選
  history:
    last_actions: [[v,omega], ...]
    last_fail_tags: [string, ...]
```

## 記憶輸入（L2 memory）

```yaml
l2_memory:
  retrieved_cases:
    - key: {region_id, obstacle_pattern_id, approach_heading_bin}
      local_path: [[x,y], ...]
      quality: float
      safety_score: float
      success_tag: bool
```

---

## L2 輸出結構（給 L3）

```yaml
l2_plan:
  local_path: [[x,y], ...]            # N 點（主要輸出）
  path_validity_score: float
  expected_clearance: float
  target_speed_profile: [float, ...]  # 對應 local_path
  fallback_mode: none|slowdown|reroute
  planner_confidence: float
```

> 關鍵：L2 不直接輸出最終控制 command；L3 只負責 follow `local_path + speed_profile`。

---

## 成功判準（先可行，再最優）
1) feasible_path_found_rate ↑
2) collision_rate ↓
3) min_clearance(p10/p50) ↑
4) success_rate 不下降（再追求上升）
