from __future__ import annotations

import argparse
from typing import Any

from .common import add_common_io_args, finalize_output, get_attr_path, load_msg_class, load_yaml, maybe_load_ros, tool_result
from .metrics_core import summarize_pose_jitter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 pose jitter evaluator (static 60s window).")
    add_common_io_args(p)
    p.add_argument("--duration-sec", type=float, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    wp0 = cfg["wp0"]
    pj = wp0["pose_jitter"]
    thresholds = wp0["thresholds"]
    duration_sec = float(args.duration_sec or wp0.get("window_sec", 60.0))
    std_limit_m = float(thresholds["pose_jitter_std_m"])

    topic = pj["topic"]
    msg_type = load_msg_class(pj["type"])
    pose_field = pj.get("pose_field", "pose")
    header_field = pj.get("header_field", "header")
    req_frame = pj.get("required_frame_id", "world")

    rclpy, Node = maybe_load_ros()

    class PoseNode(Node):
        def __init__(self) -> None:
            super().__init__("v5_wp0_pose_jitter_eval")
            self.points: list[list[float]] = []
            self.frame_mismatch = 0
            self.samples = 0
            self.sub = self.create_subscription(msg_type, topic, self.cb, 50)

        def cb(self, msg: Any) -> None:
            self.samples += 1
            try:
                header = get_attr_path(msg, header_field)
                if getattr(header, "frame_id", None) not in (None, "", req_frame):
                    self.frame_mismatch += 1
            except Exception:
                pass
            try:
                pose = get_attr_path(msg, pose_field)
                pos = pose.position
                self.points.append([float(pos.x), float(pos.y), float(pos.z)])
            except Exception:
                pass

    rclpy.init()
    node = PoseNode()
    try:
        end_time = node.get_clock().now().nanoseconds + int(duration_sec * 1e9)
        while rclpy.ok() and node.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(node, timeout_sec=0.2)

        metrics = summarize_pose_jitter(node.points, std_limit_m=std_limit_m)
        metrics.update(
            {
                "window_sec": duration_sec,
                "topic": topic,
                "required_frame_id": req_frame,
                "observed_samples": node.samples,
                "frame_mismatch_samples": node.frame_mismatch,
                "auxiliary": {"radial_std_definition": "std(norm(p - mean(p)))"},
            }
        )
        ok = bool(metrics.get("gate", {}).get("pass", False)) and node.frame_mismatch == 0
        if node.frame_mismatch:
            metrics.setdefault("warnings", []).append("frame_mismatch_detected")
        out = tool_result("pose_jitter_eval", cfg, metrics, ok=ok)
        finalize_output(out, args)
        return 0 if out.get("ok") else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
