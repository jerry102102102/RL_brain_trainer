# Phase 1 Approach -> Dock Closeout

Purpose: summarize the current Phase 1 kinematic manipulation result after the
Approach/Dock split, including what is confirmed, what remains local to the
current curriculum workspace, and which artifacts/configurations should be
treated as the current baseline.

## Scope

This document covers the pure kinematic Gymnasium/SB3 Phase 1 line. It does not
claim Gazebo, ROS, real robot, tray dynamics, or contact-rich deployment
success. The current result is a kinematic end-to-end skill result.

## Current System

The maintained Phase 1 control decomposition is:

```text
joint state + target pose
  -> Approach policy
  -> confirmed handoff state
  -> Dock policy
  -> strict local docking / hold
```

The split is intentional:

- Approach is the dynamic, larger-range policy. Its job is to move through the
  workspace and produce a clean handoff state.
- Dock is the static, local precision policy. Its job is not to generalize over
  the whole workspace; its job is to preserve, hold, and make small corrections.

This division was kept because combining large-range motion and final precision
in one policy made the final hold behavior less reliable.

## Current Best Baseline

Current best local/curriculum end-to-end baseline:

- Approach checkpoint:
  `artifacts/kinematic_phase1/phase1c/approach_finisher_ready_v2_settle_ft_786k_001/model_latest.zip`
- Dock checkpoint:
  `artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip`
- Approach config:
  `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/approach_finisher_ready_v2_settle.yaml`
- Dock config:
  `hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/dock_workspace_handoff_noop_ft_12env.yaml`

The model checkpoints are local artifacts and intentionally not tracked in git.
The code, configs, and evaluation tools needed to reproduce the line are tracked.

## Confirmed Results

The best Approach -> Dock pipeline was evaluated across six curriculum regions
using first-confirmed handoff:

| Stage | Region | Success | Handoff pos | Handoff ori | Final pos | Final ori | Final action | Final dq |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | region_small | 1.00 | 0.50 mm | 0.0073 rad | 1.67 mm | 0.0108 rad | 0.169 | 0.00049 |
| 1 | region_medium | 1.00 | 0.62 mm | 0.0099 rad | 1.67 mm | 0.0125 rad | 0.169 | 0.00049 |
| 2 | region_medium_wide | 1.00 | 0.85 mm | 0.0119 rad | 1.82 mm | 0.0141 rad | 0.169 | 0.00049 |
| 3 | region_large | 1.00 | 1.20 mm | 0.0138 rad | 2.14 mm | 0.0166 rad | 0.169 | 0.00049 |
| 4 | region_large_offset | 1.00 | 1.71 mm | 0.0150 rad | 2.53 mm | 0.0166 rad | 0.169 | 0.00049 |
| 5 | region_wide_local_random | 0.93 | 1.96 mm | 0.0177 rad | 2.89 mm | 0.0210 rad | 0.166 | 0.00051 |

Interpretation:

- The two-policy pipeline is stable across the current curriculum workspace.
- Dock action and joint motion were reduced substantially compared with earlier
  Dock variants.
- Final position is still slightly worse than handoff position, but the absolute
  error remains in the few-millimeter range and is currently acceptable for the
  intended kinematic Phase 1 tolerance.

## What Worked

The strongest recipe was not a single monolithic policy. The useful progression
was:

1. Train Approach to produce a clean handoff state.
2. Train Dock first as a local hold/capture policy.
3. Fine-tune Dock on real Approach handoff states.
4. Add no-op/hold pressure so Dock does not over-correct already-good states.

The important result is that Dock became quieter without losing end-to-end
success in the curriculum workspace.

## What Did Not Work

Several variants were informative but not kept as the current best baseline:

- A too-strict static-hold Dock from scratch entered the strict pose region but
  failed to hold it, producing large regression and large actions.
- Dock-Coarse helped in some intermediate experiments but became unnecessary
  once Approach produced sufficiently clean handoff states.
- Reward-only pressure to make Dock perfectly no-op could reduce action/dq, but
  it did not guarantee final position would be lower than the handoff position.

## Current Best Conclusion

Phase 1 is successful within the current kinematic curriculum workspace:

- Approach can reliably produce dockable handoff states.
- Dock can reliably receive those states.
- The Approach -> Dock pipeline reaches practical success rates across the
  current curriculum regions.
- Remaining work is precision refinement and expansion beyond the current
  curriculum workspace, not basic pipeline viability.

The next research question is no longer whether the two-policy decomposition can
work. It is how far the workspace can be expanded before Approach handoff quality
or Dock local precision becomes the limiting factor.
