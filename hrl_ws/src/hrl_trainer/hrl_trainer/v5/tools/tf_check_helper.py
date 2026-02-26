from __future__ import annotations

import argparse
from typing import Any

from .common import add_common_io_args, finalize_output, load_yaml, run_subprocess, tool_result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 TF check helper (view_frames + tf2_echo).")
    add_common_io_args(p)
    p.add_argument("--timeout-sec", type=float, default=5.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    wp0 = cfg.get("wp0", {})
    tf_cfg = wp0.get("tf_checks", {})
    checks = tf_cfg.get("required_pairs", [])

    results: list[dict[str, Any]] = []
    results.append(run_subprocess(["ros2", "run", "tf2_tools", "view_frames"], timeout_sec=args.timeout_sec))
    for pair in checks:
        source = pair["source"]
        target = pair["target"]
        # tf2_echo is continuous; use OS timeout to sample one successful lookup.
        cmd = ["timeout", str(int(max(1, args.timeout_sec))), "ros2", "run", "tf2_ros", "tf2_echo", source, target]
        results.append(run_subprocess(cmd, timeout_sec=args.timeout_sec + 1.0))

    ok = all(r.get("success") for r in results)
    out = tool_result("tf_check_helper", cfg, {"checks": results, "required_pairs": checks}, ok=ok)
    finalize_output(out, args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
