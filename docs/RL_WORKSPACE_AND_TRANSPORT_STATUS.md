# RL Workspace and Scene Transport Status

Purpose: This document records the current state of the RL-controlled arm after the Phase 1 / Phase 3A experiments. It is meant to clarify what the trained RL policies can and cannot currently do, especially after discovering that the reliable learned workspace is much smaller than expected.

## 0. Latest Route Curriculum Update

The route curriculum branch has now changed the status materially. The earlier policy was only a local precision controller with a full-route longest prefix of 21 waypoints. After adding route-specific observation keys and sequential actual-final-q training, the best route-trained policy is now:

```text
best current route checkpoint:
artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip

validated prefix:
prefix_120 sequential actual-final-q eval

success rate: 1.0
longest success prefix: 120
cumulative successful route distance: 1.720 m
mean final position error: 0.00934 m
mean final orientation error: 0.02444 rad
```

The same prefix_120 model was also probed on the full 483-waypoint route:

```text
full 483-waypoint probe using prefix_120 model:
success rate: 0.4741
longest success prefix: 170
cumulative successful route distance: 2.455 m
first failure index: 171
first failure reason: position
```

This means route curriculum training is working. The project is no longer stuck at the first 20-21 dense waypoints. However, the full holder1-to-holder8 route is still not solved.

Two prefix_180 attempts must be preserved as failed directions. A standard prefix_180 fine-tune regressed badly under sequential prefix evaluation:

```text
prefix_180 fine-tune:
success rate: 0.5667 over indices 1-180
longest success prefix: 1
first failure index: 2
first failure reason: position
```

An anti-forgetting prefix_180 retry was then run from the same prefix_120 base with more route-start / earlier-prefix replay:

```text
prefix_180 anti-forgetting retry:
success rate: 0.6000 over indices 1-180
longest success prefix: 1
first failure index: 2
first failure reason: position
```

Interpretation: simply changing reset ratios and continuing PPO fine-tuning is not enough. The prefix_180 attempts modify the early route-start behavior enough that index 2 no longer satisfies the route-ready position gate, even though sampled training metrics look strong. The current best model is therefore still the prefix_120 route-observation model, not either prefix_180 model.

## 1. Current Mainline

The current intended control stack is:

```text
High-level task / VLM
-> waypoint or target specification
-> RL Approach policy
-> RL Finisher policy
-> ROS2 / Gazebo execution
```

The project goal is still to use RL to control the arm, not to replace the core controller with IK. IK has only been used as a diagnostic or target-generation tool for route construction.

The currently tested route is the tray transport route from holder1 toward holder8. A dense route was generated as joint-space targets, then used to evaluate whether the RL policy can follow many small waypoints across the full scene-level transport path.

## 2. Important Correction: The Learned Workspace Is Not Yet Large

Earlier Phase 1 results showed strong local precision, but the reliable learned workspace was smaller than assumed.

The best visible-workspace training region reached approximately:

```text
x span: ~0.47 m
y span: ~0.68 m
z span: ~0.068 m
start-to-goal p90 distance: ~0.258 m
```

However, the scene-level tray transport route is much larger:

```text
x span: ~0.81 m
y span: ~1.44 m
z span: ~0.74 m
full dense route cumulative path length: ~5.84 m
```

The sampled physical FK workspace is larger than the learned policy workspace. Therefore, the limitation is not only physical reachability. The main limitation is that the RL policy has not yet learned robust control over the full scene-level route distribution.

## 3. What the Current RL Policy Can Do

The current RL policy can act as a local correction / local servo controller.

In small local waypoint tracking tests near the route start:

```text
GZ q_goal probe, first 20 dense waypoints:
target count: 20
handoff confirmed rate: 1.0
mean final position error: 0.00234 m
mean final orientation error: 0.0630 rad
strict success rate: 0.15
```

This shows that when consecutive targets are very close and remain in a familiar local distribution, the RL controller can move the arm accurately, with millimeter-level position error.

This does not prove full transport ability. It only proves local waypoint correction ability.

## 4. What the Current RL Policy Cannot Yet Do

The current RL policy cannot reliably follow the full scene-level transport route.

The most important numeric test was:

```text
Full dense route numeric validation
mode: sequential actual final_q -> next dense q_goal
target count: 483
success rate: 0.0435
finisher-ready hit rate: 0.0062
finisher-ready dwell rate: 0.0
longest continuous success prefix: 21 waypoints
mean final position error: 0.773 m
mean final orientation error: 2.44 rad
```

The longest successful prefix covers only:

```text
~1.65 cm cumulative route distance
```

The full dense route is:

```text
~5.84 m cumulative route distance
```

This means the current policy does not yet scale from local correction to full route transport.

## 5. Where the Full Route Fails

The first failure occurs at route index 22:

```text
position error: 0.00532 m
orientation error: 0.110 rad
minimum position error: 0.00457 m
minimum orientation error: 0.0807 rad
```

This is informative: the first failure is not a catastrophic position failure. It first misses because the orientation gate is not satisfied. After that, errors accumulate and the policy quickly leaves the reliable local distribution.

Chunk-level breakdown:

```text
route indices 1-40:
success rate: 0.525
mean final position error: 0.045 m
mean final orientation error: 0.408 rad

route indices 41-80:
success rate: 0.0
mean final position error: 0.714 m
mean final orientation error: 2.66 rad

route indices 81-120:
success rate: 0.0
mean final position error: 0.748 m
mean final orientation error: 2.82 rad
```

Once the policy falls off the early route prefix, it does not recover.

## 6. Scene Pose Targets vs Joint-Space Route Targets

Two target representations were tested:

### Dense scene pose targets

These are interpolated task-space pose targets.

```text
GZ dense pose probe, 5 targets:
success rate: 0.0
handoff confirmed rate: 0.0
mean final position error: 0.067 m
mean final orientation error: 0.296 rad
```

This was better than the old model, but still not close enough for handoff.

### Dense q_goal route targets

These are joint-space route waypoints used as target states. The RL policy still outputs actions; q_goal is not used as a controller command.

Small early-route q_goal tests worked well:

```text
GZ first 20 dense q_goal waypoints:
handoff confirmed rate: 1.0
mean final position error: 0.00234 m
```

But full-route numeric q_goal tracking failed:

```text
full dense route success rate: 0.0435
longest success prefix: 21 waypoints
```

Conclusion: q_goal waypoints make the local task easier and cleaner, but they do not by themselves solve full-route RL transport.

## 7. Why This Matters for the Demo

If the demo requires the arm to carry a tray from one holder to another across the full scene, the current RL controller is still not yet sufficient by itself. However, the route curriculum work has shown that the controller can be expanded well beyond the original local prefix.

The current policy is useful for:

- local correction
- final adjustment
- small waypoint-to-waypoint movement
- stabilizing near a known local distribution

The current best route policy is not yet reliable for:

- full holder1-to-holder8 transport
- late-route segments beyond the currently validated prefix
- preventing position drift around the first major post-prefix failure near index 171
- preserving early-prefix ability while fine-tuning on later route chunks
- full-route recovery after accumulated route error

## 8. Current Best Interpretation

The current RL system started as a precise local controller, but route curriculum training has begun converting it into a route-following controller.

This is not a failure of RL in principle. The newer route curriculum results show that explicit route observations, prefix curriculum, and sequential actual-final-q training can expand the reachable route prefix from 21 to 120 validated waypoints, and to 170 waypoints in a full-route probe.

The current unresolved problem is no longer "the policy cannot leave the first local region." It is now:

```text
How do we extend beyond prefix_120 / full-route prefix_170 without catastrophic forgetting of the early route?
```

The prefix_180 regressions suggest the next curriculum needs more than reset-ratio tuning. The route-start behavior must be protected explicitly, and checkpoint selection must be based on sequential route evaluation rather than sampled training success alone.

A teacher-anchored segment121-180 experiment was then added. It uses the prefix120 checkpoint as an official teacher and applies a small action-imitation anchor on route indices 1-120 while training the 121-180 local segment.

Short smoke result:

```text
route_segment121_180_teacheranchored_smoke_001:
prefix120 retained: yes
prefix180 longest success prefix: 170
full483 longest success prefix: 170
sequential gate: accepted
```

Longer 1M latest result:

```text
route_segment121_180_teacheranchored_1m_001:
prefix120 retained: no
prefix180 longest success prefix: 1
sequential gate: rejected
```

Interpretation: teacher anchoring is the first method that prevented immediate prefix180 collapse, but only an early checkpoint had the desired behavior. Longer PPO fine-tuning can still drift. Future runs must save best-by-sequential-gate, not just latest.

## 9. Decision Options

There are three realistic paths from here.

### Option A: Continue RL as the core route controller

Train a stronger scene-level transport curriculum.

This would require:

- keeping route-specific observation keys enabled
- continuing from the prefix_120 route-observation checkpoint
- using anti-forgetting reset mixtures for prefix_180 and later
- selecting checkpoints by sequential actual-final-q eval
- expanding to prefix_180 / prefix_260 only after early-prefix retention is preserved

This is the most aligned with the project goal and now has concrete evidence of progress.

### Option B: Use RL as local correction, with a classical waypoint follower for long transport

Use a classical or IK-based route planner for coarse transport, then use RL for local correction / final settling.

This is more likely to produce a demo quickly, but it changes the story:

```text
classical planner handles long motion
RL handles local correction and final control
```

This is still a valid hybrid architecture, but it is not pure RL transport.

### Option C: Narrow the demo to what RL currently does well

Demonstrate local RL-controlled insertion / approach / stabilization in a small workspace.

This is the most honest short-term demo if the requirement is that RL must directly control the arm.

## 10. Current Bottom Line

The current RL controller is not yet ready for full scene-level tray transport, but it has moved beyond pure local correction.

The most accurate current description is:

```text
The route-trained RL policy can follow a substantial early route prefix in numeric sequential validation, but full holder1-to-holder8 transport still fails later in the route. The next bottleneck is curriculum expansion without forgetting, not local precision.
```

The current best checkpoint for future route work is:

```text
artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip
```
