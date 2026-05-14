"""RViz marker publisher for the Phase 3A live demo.

The marker node is intentionally presentation-oriented: it visualizes the
semantic target, pipeline label, and optional route prefix waypoints without
owning any low-level control.  The live demo uses it to make the L1 target
visible while the RL runtime drives the Gazebo arm through L2/L3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _target_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    skill = payload.get("skill_request", payload)
    target_pose = skill.get("target_pose") if isinstance(skill, Mapping) else None
    if not isinstance(target_pose, Mapping):
        intent_resolution = payload.get("intent_resolution")
        intent_packet = intent_resolution.get("intent_packet") if isinstance(intent_resolution, Mapping) else None
        candidates = intent_packet.get("place_pose_candidates") if isinstance(intent_packet, Mapping) else None
        if isinstance(candidates, list) and candidates:
            target_pose = candidates[0]
    if not isinstance(target_pose, Mapping):
        raise ValueError("Could not find target_pose in skill request")
    xyz = target_pose.get("xyz")
    rpy = target_pose.get("rpy")
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise ValueError("target_pose.xyz must be a length-3 list")
    if not isinstance(rpy, list) or len(rpy) != 3:
        raise ValueError("target_pose.rpy must be a length-3 list")
    return {"xyz": [float(v) for v in xyz], "rpy": [float(v) for v in rpy]}


def _intent_from_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    intent_resolution = payload.get("intent_resolution")
    intent_packet = intent_resolution.get("intent_packet") if isinstance(intent_resolution, Mapping) else payload.get("intent_packet")
    intent_packet = intent_packet if isinstance(intent_packet, Mapping) else {}
    skill = payload.get("skill_request", {})
    return {
        "object_id": str(intent_packet.get("object_id", "tray1")),
        "source_slot": str(intent_packet.get("source_slot", "shelf_A1")),
        "target_slot": str(intent_packet.get("target_slot", "shelf_B1")),
        "pipeline": str(skill.get("pipeline", "APPROACH -> FINISHER")) if isinstance(skill, Mapping) else "APPROACH -> FINISHER",
    }


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    import math

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _make_marker_array(
    *,
    payload: Mapping[str, Any],
    frame_id: str,
    marker_topic: str,
    route_markers_json: str | None = None,
) -> Any:
    del marker_topic
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    target = _target_from_payload(payload)
    intent = _intent_from_payload(payload)
    x, y, z = target["xyz"]
    roll, pitch, yaw = target["rpy"]
    qx, qy, qz, qw = _quat_from_rpy(roll, pitch, yaw)

    markers = MarkerArray()

    cube = Marker()
    cube.header.frame_id = frame_id
    cube.ns = "phase3a_target"
    cube.id = 1
    cube.type = Marker.CUBE
    cube.action = Marker.ADD
    cube.pose.position.x = x
    cube.pose.position.y = y
    cube.pose.position.z = z
    cube.pose.orientation.x = qx
    cube.pose.orientation.y = qy
    cube.pose.orientation.z = qz
    cube.pose.orientation.w = qw
    cube.scale.x = 0.36
    cube.scale.y = 0.36
    cube.scale.z = 0.14
    cube.color = ColorRGBA(r=0.05, g=0.75, b=1.0, a=0.55)
    markers.markers.append(cube)

    sphere = Marker()
    sphere.header.frame_id = frame_id
    sphere.ns = "phase3a_target"
    sphere.id = 2
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose.position.x = x
    sphere.pose.position.y = y
    sphere.pose.position.z = z
    sphere.pose.orientation.w = 1.0
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12
    sphere.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.9)
    markers.markers.append(sphere)

    arrow = Marker()
    arrow.header.frame_id = frame_id
    arrow.ns = "phase3a_target"
    arrow.id = 3
    arrow.type = Marker.ARROW
    arrow.action = Marker.ADD
    arrow.pose.position.x = x
    arrow.pose.position.y = y
    arrow.pose.position.z = z
    arrow.pose.orientation.x = qx
    arrow.pose.orientation.y = qy
    arrow.pose.orientation.z = qz
    arrow.pose.orientation.w = qw
    arrow.scale.x = 0.52
    arrow.scale.y = 0.065
    arrow.scale.z = 0.065
    arrow.color = ColorRGBA(r=1.0, g=0.25, b=0.15, a=0.9)
    markers.markers.append(arrow)

    label = Marker()
    label.header.frame_id = frame_id
    label.ns = "phase3a_target"
    label.id = 4
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose.position.x = x
    label.pose.position.y = y
    label.pose.position.z = z + 0.36
    label.pose.orientation.w = 1.0
    label.scale.z = 0.105
    label.color = ColorRGBA(r=1.0, g=0.95, b=0.15, a=1.0)
    label.text = (
        f"target: {intent['target_slot']}\n"
        f"object: {intent['object_id']}\n"
        f"pipeline: {intent['pipeline']}"
    )
    markers.markers.append(label)

    if route_markers_json:
        route_payload = _load_json(route_markers_json)
        points = route_payload.get("points", [])
        if isinstance(points, list) and points:
            line = Marker()
            line.header.frame_id = frame_id
            line.ns = "phase3a_route"
            line.id = 100
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.035
            line.color = ColorRGBA(r=1.0, g=0.72, b=0.05, a=1.0)
            normalized_points: list[tuple[str, str, list[float]]] = []
            for idx, raw in enumerate(points, start=1):
                if isinstance(raw, Mapping) and "xyz" in raw:
                    p = raw["xyz"]
                    name = str(raw.get("name", f"waypoint_{idx}"))
                    description = str(raw.get("description", ""))
                else:
                    p = raw
                    name = f"waypoint_{idx}"
                    description = ""
                if isinstance(p, list) and len(p) >= 3:
                    xyz = [float(p[0]), float(p[1]), float(p[2])]
                    line.points.append(Point(x=xyz[0], y=xyz[1], z=xyz[2]))
                    normalized_points.append((name, description, xyz))
            markers.markers.append(line)

            for idx, (name, description, xyz) in enumerate(normalized_points, start=1):
                waypoint = Marker()
                waypoint.header.frame_id = frame_id
                waypoint.ns = "phase3a_route_waypoints"
                waypoint.id = 200 + idx
                waypoint.type = Marker.SPHERE
                waypoint.action = Marker.ADD
                waypoint.pose.position.x = xyz[0]
                waypoint.pose.position.y = xyz[1]
                waypoint.pose.position.z = xyz[2]
                waypoint.pose.orientation.w = 1.0
                waypoint.scale.x = waypoint.scale.y = waypoint.scale.z = 0.115
                waypoint.color = ColorRGBA(r=0.1, g=1.0, b=0.15, a=0.95)
                markers.markers.append(waypoint)

                waypoint_label = Marker()
                waypoint_label.header.frame_id = frame_id
                waypoint_label.ns = "phase3a_route_waypoint_labels"
                waypoint_label.id = 300 + idx
                waypoint_label.type = Marker.TEXT_VIEW_FACING
                waypoint_label.action = Marker.ADD
                waypoint_label.pose.position.x = xyz[0]
                waypoint_label.pose.position.y = xyz[1]
                waypoint_label.pose.position.z = xyz[2] + 0.17
                waypoint_label.pose.orientation.w = 1.0
                waypoint_label.scale.z = 0.07
                waypoint_label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                short_desc = description[:46] + ("..." if len(description) > 46 else "")
                waypoint_label.text = f"L2 wp {idx}: {name}" + (f"\n{short_desc}" if short_desc else "")
                markers.markers.append(waypoint_label)

            route_label = Marker()
            route_label.header.frame_id = frame_id
            route_label.ns = "phase3a_route"
            route_label.id = 101
            route_label.type = Marker.TEXT_VIEW_FACING
            route_label.action = Marker.ADD
            first_xyz = normalized_points[0][2]
            route_label.pose.position.x = first_xyz[0]
            route_label.pose.position.y = first_xyz[1]
            route_label.pose.position.z = first_xyz[2] + 0.36
            route_label.pose.orientation.w = 1.0
            route_label.scale.z = 0.085
            route_label.color = ColorRGBA(r=0.2, g=1.0, b=1.0, a=1.0)
            route_label.text = "mock tray route inside learned RL workspace\nL2 waypoints, L3 joint execution"
            markers.markers.append(route_label)
    return markers


def publish_target_markers(
    *,
    request_json: str | Path,
    marker_topic: str = "/v5/demo/target_marker",
    status_topic: str = "/v5/demo/status",
    frame_id: str = "world",
    duration_s: float = 120.0,
    rate_hz: float = 5.0,
    route_markers_json: str | None = None,
) -> None:
    import rclpy
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray

    payload = _load_json(request_json)
    rclpy.init(args=None)
    node = rclpy.create_node("phase3a_demo_target_marker")
    marker_pub = node.create_publisher(MarkerArray, marker_topic, 10)
    status_pub = node.create_publisher(String, status_topic, 10)
    markers = _make_marker_array(
        payload=payload,
        frame_id=frame_id,
        marker_topic=marker_topic,
        route_markers_json=route_markers_json,
    )
    period = 1.0 / max(0.1, float(rate_hz))
    deadline = time.monotonic() + max(0.1, float(duration_s))
    status = String()
    status.data = "TARGET_MARKER_PUBLISHED"
    try:
        while time.monotonic() < deadline and rclpy.ok():
            now = node.get_clock().now().to_msg()
            for marker in markers.markers:
                marker.header.stamp = now
            marker_pub.publish(markers)
            status_pub.publish(status)
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(period)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish Phase 3A target markers for RViz")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--marker-topic", default="/v5/demo/target_marker")
    parser.add_argument("--status-topic", default="/v5/demo/status")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--route-markers-json", default=None)
    args = parser.parse_args()
    publish_target_markers(
        request_json=args.request_json,
        marker_topic=args.marker_topic,
        status_topic=args.status_topic,
        frame_id=args.frame_id,
        duration_s=args.duration,
        rate_hz=args.rate_hz,
        route_markers_json=args.route_markers_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
