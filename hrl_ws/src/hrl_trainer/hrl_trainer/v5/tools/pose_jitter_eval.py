from __future__ import annotations

import argparse
import time
from typing import Any

from .common import add_common_io_args, finalize_output, get_attr_path, load_msg_class, load_yaml, maybe_load_ros, tool_result
from .metrics_core import summarize_pose_jitter

FORCED_JITTER_TOPIC = "/tray4/pose"
FORCED_JITTER_TYPE = "geometry_msgs/msg/PoseStamped"
FORCED_WORLD_FRAME = "world"
EXCLUDED_JITTER_TOPICS = {"/tray_tracking/pose_stream"}


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

    topic = FORCED_JITTER_TOPIC
    msg_type_name = FORCED_JITTER_TYPE
    pose_field = pj.get("pose_field", "pose")
    header_field = pj.get("header_field", "header")
    req_frame = FORCED_WORLD_FRAME
    req_frame_candidates = [req_frame, *(pj.get("required_frame_id_candidates") or [])]
    req_frame_candidates = [x for x in req_frame_candidates if isinstance(x, str) and x]
    source_specs = [{"topic": topic, "type": msg_type_name, "pose_field": pose_field, "header_field": header_field}]
    cfg_excluded = {str(x) for x in (pj.get("excluded_topics") or []) if isinstance(x, str)}
    excluded_topics = sorted(EXCLUDED_JITTER_TOPICS.union(cfg_excluded))
    warnings: list[str] = []
    if str(pj.get("topic", "")) not in ("", FORCED_JITTER_TOPIC):
        warnings.append("pose_jitter_topic_overridden_to_tray4_pose")
    if str(pj.get("type", "")) not in ("", FORCED_JITTER_TYPE):
        warnings.append("pose_jitter_type_overridden_to_posestamped")
    if str(pj.get("required_frame_id", "")) not in ("", FORCED_WORLD_FRAME):
        warnings.append("pose_jitter_required_frame_overridden_to_world")

    rclpy, Node = maybe_load_ros()

    class PoseNode(Node):
        def __init__(self) -> None:
            super().__init__("v5_wp0_pose_jitter_eval")
            self.set_parameters([
                rclpy.parameter.Parameter(
                    "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, bool(wp0.get("use_sim_time", False))
                )
            ])
            self.points: list[list[float]] = []
            self.frame_mismatch = 0
            self.samples = 0
            topic_types = {name: list(types) for name, types in self.get_topic_names_and_types()}
            self.selected_source = source_specs[0]
            self.selected_topic_types = topic_types.get(str(self.selected_source["topic"]), [])
            self.selected_topic = str(self.selected_source["topic"])
            self.selected_type = str(self.selected_source["type"])
            self.selected_mode = "pose"
            self.selected_pose_field = str(self.selected_source.get("pose_field", pose_field))
            self.selected_header_field = str(self.selected_source.get("header_field", header_field))
            msg_cls = load_msg_class(self.selected_type)
            self.sub = self.create_subscription(msg_cls, self.selected_topic, self.cb, 50)

        def _frame_mismatch(self, frame_id: str | None) -> bool:
            if frame_id in (None, ""):
                return False
            return frame_id not in req_frame_candidates

        def cb(self, msg: Any) -> None:
            self.samples += 1
            try:
                header = get_attr_path(msg, self.selected_header_field)
                frame_id = getattr(header, "frame_id", None)
                if self._frame_mismatch(frame_id):
                    self.frame_mismatch += 1
            except Exception:
                pass
            try:
                pose = get_attr_path(msg, self.selected_pose_field)
                pos = pose.position
                self.points.append([float(pos.x), float(pos.y), float(pos.z)])
            except Exception:
                pass

    rclpy.init()
    node = PoseNode()
    try:
        deadline = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)

        metrics = summarize_pose_jitter(node.points, std_limit_m=std_limit_m)
        metrics.update(
            {
                "window_sec": duration_sec,
                "topic": node.selected_topic,
                "topic_types": node.selected_topic_types,
                "source_type": node.selected_type,
                "source_mode": node.selected_mode,
                "source_candidates": source_specs,
                "excluded_topics": excluded_topics,
                "required_frame_id": req_frame,
                "required_frame_id_candidates": req_frame_candidates,
                "observed_samples": node.samples,
                "frame_mismatch_samples": node.frame_mismatch,
                "auxiliary": {"radial_std_definition": "std(norm(p - mean(p)))"},
            }
        )
        ok = bool(metrics.get("gate", {}).get("pass", False)) and node.frame_mismatch == 0
        if node.frame_mismatch:
            metrics.setdefault("warnings", []).append("frame_mismatch_detected")
        if warnings:
            metrics.setdefault("warnings", []).extend(warnings)
        out = tool_result("pose_jitter_eval", cfg, metrics, ok=ok)
        finalize_output(out, args)
        return 0 if out.get("ok") else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
