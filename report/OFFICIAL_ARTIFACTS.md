# Official Artifacts and Numbers

Purpose: This is the single source of truth for final report, slides, demo scripts, and README claims.

## A. Phase 1 Baseline

Phase 1 was operational but incomplete. The custom Gymnasium kinematic environment, training loop, throughput checks, and deterministic evaluation tools were running. Isolated approach and dock policies improved, but switched integration was unreliable and the VLM/Qwen layer had not started.

## B. Phase 2 Skill Stack

Final skill path:

```text
Approach -> Finisher
```

Dock-Coarse, Bridge, readiness classifier, acceptance map, and finisher adaptation are diagnostic steps. They helped identify the clean final path, but they are not the main final controller.

| Stage | Success | Handoff Pos Error | Handoff Ori Error | Final Pos Error | Final Ori Error |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.50 mm | 0.0073 rad | 1.67 mm | 0.0106 rad |
| 1 | 1.00 | 0.62 mm | 0.0099 rad | 1.67 mm | 0.0123 rad |
| 2 | 1.00 | 0.85 mm | 0.0119 rad | 1.82 mm | 0.0139 rad |
| 3 | 1.00 | 1.20 mm | 0.0138 rad | 2.14 mm | 0.0164 rad |
| 4 | 1.00 | 1.71 mm | 0.0150 rad | 2.53 mm | 0.0165 rad |
| 5 | 0.93 | 1.96 mm | 0.0177 rad | 2.89 mm | 0.0208 rad |

Core Phase 2 result:

```text
Stage 5 success rate: 0.93
Stage 5 handoff position error: 1.96 mm
Stage 5 handoff orientation error: 0.0177 rad
Stage 5 final position error: 2.89 mm
Stage 5 final orientation error: 0.0208 rad
```

## C. Qwen L1 Bridge

Demo command:

```text
Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose.
```

Tool call:

```json
{
  "arguments": {
    "constraints": {
      "speed_cap": "SLOW"
    },
    "object_id": "tray1",
    "source_slot": "shelf_A1",
    "target_slot": "shelf_B1"
  },
  "tool": "resolve_intent_packet"
}
```

Resolved skill request:

```text
object_id: tray1
source_slot: shelf_A1
target_slot: shelf_B1
pipeline: APPROACH -> FINISHER
target xyz: [-0.92, -1.16, 1.22]
target rpy: [3.14, 0.0, 3.14]
```

Safety boundary:

```text
Qwen / L1 may produce semantic intent and structured skill requests.
Qwen / L1 must not produce raw joint actions, trajectories, torques, or delta_q.
L2/L3 own policy rollout, execution, and safety.
```

## D. Route Curriculum Result

Baseline:

```text
full483 baseline success rate: 0.0435
baseline longest prefix: 21
```

Official route checkpoint:

```text
artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip
```

Best route result:

```text
prefix120 sequential success: 1.0
prefix120 longest prefix: 120
prefix120 route distance: 1.720 m
prefix120 mean final position error: 0.00934 m
prefix120 mean final orientation error: 0.02444 rad
full483 probe success: 0.4741
full483 longest prefix: 170
full483 route distance: 2.455 m
first failure index: 171
first failure reason: position
```

## E. Failed Directions

These are important research findings, not final claims:

- Prefix180 direct fine-tune failed sequential retention.
- Prefix180 anti-forgetting reset-ratio retry failed sequential retention.
- Prefix180 teacher-anchor smoke was promising and reached prefix170, but the longer 1M latest checkpoint drifted.
- Dock-Coarse as a fixed middle layer was removed from the final main path.
- Full holder1-to-holder8 transport is not solved.

## F. Workspace Expansion

The Stage 0-5 result proves a reliable local kinematic workspace, not the whole arm workspace. Later workspace-expansion experiments pushed the policy into larger manually defined stress shells.

Current useful home-start expansion checkpoint:

```text
artifacts/kinematic_phase1/workspace_expansion/workspace_expand_stage10_11_ppo_1h_002/best_checkpoint/model_best_by_gate.zip
```

Representative home-start stage result:

| Stage | Success | Final Pos Error | Final Ori Error |
|---:|---:|---:|---:|
| 0 | 1.00 | 1.63 mm | 0.0130 rad |
| 1 | 1.00 | 1.72 mm | 0.0133 rad |
| 2 | 1.00 | 1.85 mm | 0.0134 rad |
| 3 | 1.00 | 2.16 mm | 0.0141 rad |
| 4 | 0.98 | 2.49 mm | 0.0148 rad |
| 5 | 0.98 | 2.77 mm | 0.0162 rad |
| 6 | 0.93 | 2.99 mm | 0.0186 rad |
| 7 | 0.83 | 3.57 mm | 0.0201 rad |
| 8 | 0.65 | 4.16 mm | 0.0218 rad |
| 9 | 0.45 | 5.65 mm | 0.0260 rad |
| 10 | 0.40 | 6.14 mm | 0.0319 rad |
| 11 | 0.29 | 9.10 mm | 0.0446 rad |

Important interpretation:

```text
Stage 10/11 are larger evaluation shells, not the full continuous workspace.
```

## G. Full Workspace Random-Start Coverage

The random-start / mixed-start experiment tests whether the policy can start away from home and move to many reachable targets.

Recommended artifact:

```text
artifacts/kinematic_phase1/workspace_full_coverage_randomstart/workspace_full_coverage_randomstart_overnight_003/best_checkpoint/model_best_by_gate.zip
```

Key coverage result:

| Eval Split | Success | Mean Final Pos Error | Mean Final Ori Error | Main Failure |
|---|---:|---:|---:|---|
| Known workspace random-start | 0.802 | 3.87 mm | 0.0185 rad | position / timeout |
| Frontier random-start | 0.240 | 260 mm | 0.611 rad | position |
| Full reachable stress | 0.219 | 424 mm | 1.156 rad | position |

Coverage summary:

```text
covered_bucket_fraction: 0.20
stable_bucket_fraction: 0.0286
partial_bucket_fraction: 0.1714
stress_bucket_fraction: 0.80
```

Interpretation:

```text
The policy is no longer only a home-start controller inside the known region.
It has useful mixed-start behavior in known workspace, but full reachable
workspace coverage is still future work.
```
