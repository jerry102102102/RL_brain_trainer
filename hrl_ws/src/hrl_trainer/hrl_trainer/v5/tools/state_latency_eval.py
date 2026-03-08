from __future__ import annotations

import argparse
from collections import defaultdict
import time
from typing import Any

from .common import add_common_io_args, finalize_output, get_attr_path, load_msg_class, load_yaml, maybe_load_ros, stamp_to_ns, tool_result
from .metrics_core import summarize_state_topic_latency_by_topic


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 state-topic one-hop latency evaluator (recv - header.stamp).")
    add_common_io_args(p)
    p.add_argument("--duration-sec", type=float, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    wp0 = cfg["wp0"]
    duration_sec = float(args.duration_sec or wp0.get("window_sec", 60.0))
    state_topics = wp0.get("state_topics", [])
    p95_limit_ms = float(wp0["thresholds"]["state_latency_p95_ms"])

    rclpy, Node = maybe_load_ros()

    class LatNode(Node):
        def __init__(self) -> None:
            super().__init__("v5_wp0_state_latency_eval")
            self.set_parameters([
                rclpy.parameter.Parameter(
                    "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, bool(wp0.get("use_sim_time", False))
                )
            ])
            self.lats: dict[str, list[float]] = defaultdict(list)
            self.subs = []
            for spec in state_topics:
                msg_type = load_msg_class(spec["type"])
                self.subs.append(self.create_subscription(msg_type, spec["topic"], self._cb(spec), 50))

        def _cb(self, spec: dict[str, Any]):
            topic = spec["topic"]
            header_field = spec.get("header_field", "header")

            def cb(msg: Any) -> None:
                ros_recv_ns = self.get_clock().now().nanoseconds
                wall_recv_ns = time.time_ns()
                try:
                    header = get_attr_path(msg, header_field)
                    msg_ns = stamp_to_ns(header.stamp)
                except Exception:
                    return
                recv_ns = ros_recv_ns if abs(ros_recv_ns - msg_ns) <= abs(wall_recv_ns - msg_ns) else wall_recv_ns
                self.lats[topic].append((recv_ns - msg_ns) / 1e6)

            return cb

    rclpy.init()
    node = LatNode()
    try:
        deadline = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        metrics = summarize_state_topic_latency_by_topic(node.lats, p95_limit_ms=p95_limit_ms)
        metrics.update(
            {
                "window_sec": duration_sec,
                "latency_definition": "recv_time - msg.header.stamp (same ROS time basis)",
            }
        )
        total_samples = int(metrics.get("overall", {}).get("count", 0) or 0)
        if total_samples == 0:
            subscribed_topics = [str(spec.get("topic", "")) for spec in state_topics]
            metrics["diagnostics"] = {
                "reason": "no_samples_received",
                "subscribed_topics": subscribed_topics,
                "hint": "No messages were received. Verify publishers are active, especially /joint_states if expected.",
            }
        ok = bool(metrics.get("overall", {}).get("gate", {}).get("pass", False))
        out = tool_result("state_latency_eval", cfg, metrics, ok=ok)
        finalize_output(out, args)
        return 0 if ok else 1
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
