"""Phase 3A Gazebo validation runtime path.

This module bridges the Phase 2 Qwen L1 output into a safe Phase 3A runtime
request.  It is intentionally conservative: by default it validates model
assets and prepares/publishes high-level L1/L2 messages, but it does not publish
raw joint trajectories.  L3 remains the owner of executor-level control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from .intent_layer import validate_intent_packet
from .runtime_model_registry import Phase3ARuntimeRegistry, load_phase3a_model_registry


REPO_ROOT = Path(__file__).resolve().parents[5]


class Phase3ARuntimeError(ValueError):
    """Raised when a Phase 3A runtime request is invalid."""


@dataclass(frozen=True)
class Phase3ARequest:
    intent_packet: dict[str, Any]
    skill_request: dict[str, Any]
    source_path: str | None = None


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase3ARuntimeError("runtime request JSON must contain an object")
    return payload


def parse_phase3a_request(payload: Mapping[str, Any], *, source_path: str | None = None) -> Phase3ARequest:
    """Accept either a full qwen_l1_client artifact or a direct skill request."""
    if "intent_resolution" in payload and "skill_request" in payload:
        intent_resolution = payload.get("intent_resolution")
        if not isinstance(intent_resolution, Mapping):
            raise Phase3ARuntimeError("intent_resolution must be an object")
        intent_packet = intent_resolution.get("intent_packet")
        skill_request = payload.get("skill_request")
    else:
        intent_packet = payload.get("intent_packet")
        skill_request = payload

    if not isinstance(intent_packet, Mapping):
        raise Phase3ARuntimeError("request is missing a valid intent_packet")
    if not isinstance(skill_request, Mapping):
        raise Phase3ARuntimeError("request is missing a valid skill_request")

    validate_intent_packet(intent_packet)
    if skill_request.get("pipeline") != "APPROACH -> FINISHER":
        raise Phase3ARuntimeError("Phase 3A only supports pipeline='APPROACH -> FINISHER'")
    if skill_request.get("dry_run") is not True:
        raise Phase3ARuntimeError("Phase 3A first runtime path expects dry_run=true skill requests")

    return Phase3ARequest(
        intent_packet=dict(intent_packet),
        skill_request=dict(skill_request),
        source_path=source_path,
    )


def _target_pose_from_request(request: Phase3ARequest) -> dict[str, Any]:
    target_pose = request.skill_request.get("target_pose")
    if not isinstance(target_pose, Mapping):
        candidates = request.intent_packet.get("place_pose_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise Phase3ARuntimeError("No target_pose or place_pose_candidates available")
        target_pose = candidates[0]
    xyz = target_pose.get("xyz")
    rpy = target_pose.get("rpy")
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise Phase3ARuntimeError("target pose xyz must be length-3 list")
    if not isinstance(rpy, list) or len(rpy) != 3:
        raise Phase3ARuntimeError("target pose rpy must be length-3 list")
    return {"xyz": [float(v) for v in xyz], "rpy": [float(v) for v in rpy]}


def build_phase3a_rollout_plan(
    request: Phase3ARequest,
    registry: Phase3ARuntimeRegistry,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    path_checks = registry.validate_paths(repo_root)
    target_pose = _target_pose_from_request(request)
    return {
        "schema_version": "v5.phase3a.rollout_plan.v1",
        "mode": "dry_run_or_ros_dry_run",
        "source_request_path": request.source_path,
        "pipeline": registry.pipeline,
        "state_machine": ["APPROACH", "FINISHER"],
        "handoff_conditions": dict(registry.handoff),
        "topics": dict(registry.ros_topics),
        "object_id": request.intent_packet["object_id"],
        "source_slot": request.intent_packet["source_slot"],
        "target_slot": request.intent_packet["target_slot"],
        "target_pose": target_pose,
        "model_assets": registry.to_dict(),
        "path_checks": path_checks,
        "safety_boundary": {
            "l1_outputs_raw_control": False,
            "runtime_publishes_joint_trajectory": False,
            "l3_owns_joint_trajectory": True,
            "note": "Phase 3A runtime publishes/validates high-level L1/L2 messages only in this first path.",
        },
    }


def try_load_sb3_policy(checkpoint: str | Path) -> dict[str, Any]:
    """Optionally validate that an SB3 checkpoint can be deserialized."""
    errors: list[str] = []
    for algo_name in ("PPO", "TD3", "SAC"):
        try:
            module = __import__("stable_baselines3", fromlist=[algo_name])
            algo = getattr(module, algo_name)
            _ = algo.load(str(checkpoint), device="cpu")
            return {"loaded": True, "algo": algo_name, "checkpoint": str(checkpoint)}
        except Exception as exc:  # pragma: no cover - depends on optional model package/assets
            errors.append(f"{algo_name}: {exc}")
    return {"loaded": False, "checkpoint": str(checkpoint), "errors": errors[-3:]}


class Phase3ARosDryRunPublisher:
    """Publish high-level contract messages without executor-level trajectory output."""

    def __init__(self, topics: Mapping[str, str]):
        try:
            import rclpy
            from std_msgs.msg import String
        except Exception as exc:  # pragma: no cover - only exercised in ROS2 env
            raise RuntimeError("ROS2 std_msgs/rclpy are required for ros_dry_run mode") from exc

        self._rclpy = rclpy
        self._String = String
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        self._node = rclpy.create_node("phase3a_runtime_node")
        self._intent_pub = self._node.create_publisher(String, topics.get("intent_packet", "/v5/intent_packet"), 10)
        self._skill_pub = self._node.create_publisher(String, topics.get("skill_command", "/v5/skill_command"), 10)

    def publish(self, *, intent_packet: Mapping[str, Any], skill_command: Mapping[str, Any], spin_s: float = 0.2) -> None:
        intent_msg = self._String()
        intent_msg.data = json.dumps(intent_packet, sort_keys=True)
        skill_msg = self._String()
        skill_msg.data = json.dumps(skill_command, sort_keys=True)
        self._intent_pub.publish(intent_msg)
        self._skill_pub.publish(skill_msg)
        deadline = time.monotonic() + max(0.0, float(spin_s))
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.02)

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()


def run_phase3a_runtime(
    *,
    request_json: str | Path,
    registry_path: str | Path | None = None,
    output_json: str | Path | None = None,
    mode: str = "dry_run",
    load_policies: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    payload = _load_json(request_json)
    request = parse_phase3a_request(payload, source_path=str(request_json))
    registry = load_phase3a_model_registry(registry_path)
    plan = build_phase3a_rollout_plan(request, registry, repo_root=repo_root)

    if load_policies:
        repo = Path(repo_root)
        plan["policy_load_checks"] = {
            "approach": try_load_sb3_policy(registry.approach.resolve_checkpoint(repo)),
            "finisher": try_load_sb3_policy(registry.finisher.resolve_checkpoint(repo)),
        }

    if mode == "ros_dry_run":
        publisher = Phase3ARosDryRunPublisher(registry.ros_topics)
        try:
            publisher.publish(intent_packet=request.intent_packet, skill_command=request.skill_request)
        finally:
            publisher.close()
        plan["ros_dry_run_published"] = True
    elif mode != "dry_run":
        raise Phase3ARuntimeError(f"Unsupported Phase 3A runtime mode: {mode}")

    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3A Approach->Finisher runtime bridge")
    parser.add_argument("--request-json", required=True, help="Qwen L1 client artifact or skill request JSON")
    parser.add_argument("--registry", default=None, help="Override phase3a_runtime_models.yaml")
    parser.add_argument("--output-json", required=True, help="Write runtime plan/status JSON")
    parser.add_argument("--mode", choices=["dry_run", "ros_dry_run"], default="dry_run")
    parser.add_argument("--load-policies", action="store_true", help="Attempt to deserialize SB3 checkpoints on CPU")
    args = parser.parse_args()

    plan = run_phase3a_runtime(
        request_json=args.request_json,
        registry_path=args.registry,
        output_json=args.output_json,
        mode=args.mode,
        load_policies=args.load_policies,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
