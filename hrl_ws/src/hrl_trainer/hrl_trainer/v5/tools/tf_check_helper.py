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
    view_frames = run_subprocess(["ros2", "run", "tf2_tools", "view_frames"], timeout_sec=args.timeout_sec)
    if not view_frames.get("success"):
        txt = f"{view_frames.get('stdout', '')}\n{view_frames.get('stderr', '')}"
        # view_frames can exceed short timeout windows after it already started listening.
        if "Listening to tf data" in txt and "PackageNotFoundError" not in txt:
            view_frames["success"] = True
            view_frames["note"] = "view_frames timed out after starting TF capture"
    results.append(view_frames)
    for pair in checks:
        source = pair["source"]
        target = pair["target"]
        # tf2_echo args are <target_frame> <source_frame>; checks are configured as source->target.
        # tf2_echo is continuous, so "timeout" exits with 124 even after successful lookups.
        cmd = ["timeout", str(int(max(1, args.timeout_sec))), "ros2", "run", "tf2_ros", "tf2_echo", target, source]
        res = run_subprocess(cmd, timeout_sec=args.timeout_sec + 1.0)
        if not res.get("success"):
            txt_out = str(res.get("stdout", ""))
            txt = f"{txt_out}\n{res.get('stderr', '')}"
            # tf2_echo is continuous and often starts before all frame ids appear.
            # If we captured at least one transform sample, treat timeout as success.
            if res.get("returncode") == 124 and "At time" in txt_out:
                res["success"] = True
                res["note"] = "tf2_echo timed out after successful transform sampling"
        results.append(res)

    ok = all(r.get("success") for r in results)
    out = tool_result("tf_check_helper", cfg, {"checks": results, "required_pairs": checks}, ok=ok)
    finalize_output(out, args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
