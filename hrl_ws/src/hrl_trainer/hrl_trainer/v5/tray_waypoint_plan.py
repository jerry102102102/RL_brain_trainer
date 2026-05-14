"""VLM-style tray waypoint planning helpers for Phase 3A.

This module is intentionally pragmatic: it turns a high-level tray carry
instruction into (1) a human-readable semantic waypoint plan and (2) a target
JSON that the existing Phase 3A controlled simulator can execute with the
trained Approach -> Finisher policies.

The first executable mode uses local FK-reachable joint deltas.  That keeps the
demo inside the currently trained RL workspace while preserving the L1 contract:
the language layer chooses a structured waypoint sequence, but L2/L3 still own
policy inference and joint-level execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SemanticWaypoint:
    name: str
    description: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    hold_level: bool = True
    phase_hint: str = "APPROACH_TO_FINISHER"

    @property
    def pose6(self) -> tuple[float, float, float, float, float, float]:
        return (*self.xyz, *self.rpy)

    def to_plan_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "phase_hint": self.phase_hint,
            "hold_level": self.hold_level,
            "posture_constraint": "EE tray plane stays horizontal to the table",
            "target_encoding": "local_level_pose6",
            "pose6": list(self.pose6),
        }

    def to_control_target(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "vlm_semantic_waypoint/local_level_pose6",
            "pose6": list(self.pose6),
        }


def default_tray_carry_waypoints() -> list[SemanticWaypoint]:
    """Return a visible local tray-carry waypoint sequence.

    These waypoints are still a mock tray route inside the learned RL workspace,
    not a claim of full holder1-to-holder8 transport.  The positions follow a
    visible source-to-destination carry motion, while every waypoint uses the
    same tray-level end-effector orientation.  That preserves the intended
    semantics: insert level, lift level, carry level, and settle level.
    """

    # This is the "tray-level" orientation used by the original controller
    # convention: EE y-axis is aligned with world +Z, so the tray plane stays
    # horizontal.  The yaw term is held fixed for this presentation route so the
    # demo does not imply uncontrolled wrist twisting while carrying.
    level_rpy = (1.5708, 0.0, -1.5708)

    return [
        SemanticWaypoint(
            name="pre_grasp_align",
            description="Move from home toward the tray approach side with the EE already level.",
            xyz=(-0.204, -0.288, 1.050),
            rpy=level_rpy,
        ),
        SemanticWaypoint(
            name="under_tray_insert_pose",
            description="Slide forward under the virtual tray while keeping the EE horizontal.",
            xyz=(-0.189, -0.162, 1.050),
            rpy=level_rpy,
        ),
        SemanticWaypoint(
            name="level_lift",
            description="Lift straight up into a carry height without tilting the tray plane.",
            xyz=(-0.180, -0.094, 1.094),
            rpy=level_rpy,
        ),
        SemanticWaypoint(
            name="carry_midline",
            description="Carry across the local workspace while holding the EE level.",
            xyz=(-0.195, 0.151, 1.094),
            rpy=level_rpy,
        ),
        SemanticWaypoint(
            name="pre_insert_align",
            description="Align with the destination insertion side while still level.",
            xyz=(-0.186, 0.194, 1.080),
            rpy=level_rpy,
        ),
        SemanticWaypoint(
            name="stable_insert_hold",
            description="Hold the final insertion pose with the EE horizontal and low motion.",
            xyz=(-0.222, 0.302, 1.050),
            rpy=level_rpy,
        ),
    ]


def build_vlm_plan(
    *,
    instruction: str,
    source_slot: str,
    target_slot: str,
    object_id: str,
    waypoints: Iterable[SemanticWaypoint],
) -> dict[str, Any]:
    rows = [wp.to_plan_row() for wp in waypoints]
    return {
        "schema_version": "phase3a.tray_waypoint_plan.v1",
        "planner": "qwen_or_llm_structured_waypoint_mock",
        "instruction": instruction,
        "object_id": object_id,
        "source_slot": source_slot,
        "target_slot": target_slot,
        "pipeline": ["APPROACH", "FINISHER"],
        "safety_boundary": {
            "l1_outputs_joint_trajectory": False,
            "l1_outputs_waypoints_only": True,
            "l2_l3_execute_policy_and_trajectory": True,
        },
        "waypoints": rows,
    }


def build_control_targets(waypoints: Iterable[SemanticWaypoint]) -> dict[str, Any]:
    return {
        "schema_version": "phase3a.controlled_targets.v1",
        "target_encoding": "pose6",
        "targets": [wp.to_control_target() for wp in waypoints],
    }


def write_outputs(
    *,
    output_dir: Path,
    instruction: str,
    source_slot: str,
    target_slot: str,
    object_id: str,
) -> Mapping[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    waypoints = default_tray_carry_waypoints()
    plan = build_vlm_plan(
        instruction=instruction,
        source_slot=source_slot,
        target_slot=target_slot,
        object_id=object_id,
        waypoints=waypoints,
    )
    targets = build_control_targets(waypoints)
    plan_path = output_dir / "vlm_tray_waypoint_plan.json"
    targets_path = output_dir / "controlled_targets.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    targets_path.write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "plan_path": str(plan_path),
        "targets_path": str(targets_path),
        "waypoint_count": str(len(waypoints)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--instruction",
        default="Move tray1 from shelf_A1 to shelf_B1 while keeping it level.",
    )
    parser.add_argument("--source-slot", default="shelf_A1")
    parser.add_argument("--target-slot", default="shelf_B1")
    parser.add_argument("--object-id", default="tray1")
    args = parser.parse_args()
    result = write_outputs(
        output_dir=Path(args.output_dir),
        instruction=args.instruction,
        source_slot=args.source_slot,
        target_slot=args.target_slot,
        object_id=args.object_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
