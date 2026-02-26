from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from .common import add_common_io_args, finalize_output, load_yaml, maybe_load_ros, stamp_to_ns, tool_result
from .metrics_core import greedy_approx_sync_pairs_ns


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 approx sync evaluator for overhead+side images.")
    add_common_io_args(p)
    p.add_argument("--duration-sec", type=float, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    wp0 = cfg["wp0"]
    thresholds = wp0["thresholds"]
    duration_sec = float(args.duration_sec or wp0.get("window_sec", 60.0))
    slop_ms = float(thresholds["approx_sync_slop_ms"])
    pass_rate = float(thresholds["approx_sync_success_rate_min"])
    queue_size = int(wp0.get("approx_sync", {}).get("queue_size", 10))

    overhead_topic = wp0["cameras"]["overhead"]["image_topic"]
    side_topic = wp0["cameras"]["side"]["image_topic"]

    rclpy, Node = maybe_load_ros()

    class SyncNode(Node):
        def __init__(self) -> None:
            super().__init__("v5_wp0_approx_sync_eval")
            from sensor_msgs.msg import Image  # type: ignore

            self.stamps: dict[str, list[int]] = defaultdict(list)
            self.subs = [
                self.create_subscription(Image, overhead_topic, self._cb(overhead_topic), queue_size),
                self.create_subscription(Image, side_topic, self._cb(side_topic), queue_size),
            ]

        def _cb(self, topic: str):
            def cb(msg: Any) -> None:
                try:
                    self.stamps[topic].append(stamp_to_ns(msg.header.stamp))
                except Exception:
                    pass

            return cb

    rclpy.init()
    node = SyncNode()
    try:
        end_time = node.get_clock().now().nanoseconds + int(duration_sec * 1e9)
        while rclpy.ok() and node.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(node, timeout_sec=0.2)

        metrics = greedy_approx_sync_pairs_ns(node.stamps.get(overhead_topic, []), node.stamps.get(side_topic, []), slop_ms=slop_ms)
        metrics["topics"] = {"overhead": overhead_topic, "side": side_topic}
        metrics["window_sec"] = duration_sec
        metrics["queue_size"] = queue_size
        metrics["gate"] = {
            "success_rate_min": pass_rate,
            "pass": bool(metrics["success_rate"] > pass_rate),
            "slop_ms": slop_ms,
        }
        out = tool_result("approx_sync_eval", cfg, metrics, ok=metrics["gate"]["pass"])
        finalize_output(out, args)
        return 0 if out.get("ok") else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
