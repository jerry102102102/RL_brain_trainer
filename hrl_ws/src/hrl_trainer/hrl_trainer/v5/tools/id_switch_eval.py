from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import add_common_io_args, finalize_output, get_attr_path, load_msg_class, load_yaml, maybe_load_ros, tool_result
from .metrics_core import summarize_id_switch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 object id-switch / missing-rate evaluator.")
    add_common_io_args(p)
    p.add_argument("--mode", choices=["jsonl", "ros"], default=None, help="Override configured input mode")
    p.add_argument("--jsonl", default=None, help="JSONL path for offline id samples")
    p.add_argument("--duration-sec", type=float, default=None, help="ROS mode only")
    return p


def _run_jsonl(cfg: dict[str, Any], args: argparse.Namespace, section: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    path = Path(args.jsonl or section.get("jsonl_path") or "")
    if not path.exists():
        metrics = {"error": f"jsonl file not found: {path}"}
        return metrics, False

    id_field = section.get("jsonl_id_field", "object_id")
    valid_field = section.get("jsonl_valid_field", "valid")
    ids: list[Any] = []
    flags: list[bool] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(get_attr_path(row, id_field) if id_field else row)
        if valid_field:
            try:
                flags.append(bool(get_attr_path(row, valid_field)))
            except Exception:
                flags.append(True)
        else:
            flags.append(True)

    warn_rate = float(cfg["wp0"]["thresholds"]["id_missing_warn_rate"])
    switch_limit = float(cfg["wp0"]["thresholds"]["id_switch_rate_max"])
    metrics = summarize_id_switch(ids, valid_flags=flags, missing_warn_rate=warn_rate)
    metrics.update(
        {
            "source": {"mode": "jsonl", "path": str(path)},
            "gate": {
                "switch_rate_max": switch_limit,
                "pass": bool(metrics.get("switch_rate") is not None and metrics["switch_rate"] < switch_limit),
            },
        }
    )
    return metrics, bool(metrics["gate"]["pass"])


def _run_ros(cfg: dict[str, Any], args: argparse.Namespace, section: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    rclpy, Node = maybe_load_ros()
    wp0 = cfg["wp0"]
    duration_sec = float(args.duration_sec or wp0.get("window_sec", 60.0))
    topic = section["topic"]
    msg_type = load_msg_class(section["type"])
    id_field = section["id_field"]
    valid_field = section.get("valid_field")
    warn_rate = float(wp0["thresholds"]["id_missing_warn_rate"])
    switch_limit = float(wp0["thresholds"]["id_switch_rate_max"])

    class IdNode(Node):
        def __init__(self) -> None:
            super().__init__("v5_wp0_id_switch_eval")
            self.ids: list[Any] = []
            self.flags: list[bool] = []
            self.sub = self.create_subscription(msg_type, topic, self.cb, 20)

        def cb(self, msg: Any) -> None:
            try:
                obj_id = get_attr_path(msg, id_field)
            except Exception:
                obj_id = None
            is_valid = True
            if valid_field:
                try:
                    is_valid = bool(get_attr_path(msg, valid_field))
                except Exception:
                    is_valid = True
            self.ids.append(obj_id)
            self.flags.append(is_valid)

    rclpy.init()
    node = IdNode()
    try:
        end_time = node.get_clock().now().nanoseconds + int(duration_sec * 1e9)
        while rclpy.ok() and node.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(node, timeout_sec=0.2)
        metrics = summarize_id_switch(node.ids, valid_flags=node.flags, missing_warn_rate=warn_rate)
        metrics.update(
            {
                "source": {"mode": "ros", "topic": topic, "type": section["type"]},
                "window_sec": duration_sec,
                "gate": {
                    "switch_rate_max": switch_limit,
                    "pass": bool(metrics.get("switch_rate") is not None and metrics["switch_rate"] < switch_limit),
                },
            }
        )
        return metrics, bool(metrics["gate"]["pass"])
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    section = cfg["wp0"]["id_switch"]
    mode = args.mode or section.get("mode", "jsonl")
    if mode == "jsonl":
        metrics, ok = _run_jsonl(cfg, args, section)
    else:
        metrics, ok = _run_ros(cfg, args, section)
    out = tool_result("id_switch_eval", cfg, metrics, ok=ok)
    finalize_output(out, args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
