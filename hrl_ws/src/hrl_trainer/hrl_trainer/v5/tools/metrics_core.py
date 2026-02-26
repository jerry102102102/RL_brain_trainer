"""Pure metric computation helpers for V5 WP0 diagnostics.

This module is intentionally ROS-free so it can be unit tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class LatencyGate:
    p95_ms_limit: float


def _clean_floats(values: Iterable[Any]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            cleaned.append(fv)
    return cleaned


def percentile_ms(values_ms: Iterable[Any], percentile: float) -> float | None:
    vals = _clean_floats(values_ms)
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), percentile))


def summarize_latency_ms(values_ms: Iterable[Any], p95_limit_ms: float | None = None) -> dict[str, Any]:
    vals = _clean_floats(values_ms)
    out: dict[str, Any] = {
        "count": len(vals),
        "p50_ms": None,
        "p95_ms": None,
        "max_ms": None,
        "mean_ms": None,
    }
    if not vals:
        if p95_limit_ms is not None:
            out["gate"] = {"p95_ms_limit": p95_limit_ms, "pass": False, "reason": "no_samples"}
        return out

    arr = np.asarray(vals, dtype=float)
    out.update(
        {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(np.max(arr)),
            "mean_ms": float(np.mean(arr)),
        }
    )
    if p95_limit_ms is not None:
        out["gate"] = {
            "p95_ms_limit": float(p95_limit_ms),
            "pass": bool(out["p95_ms"] is not None and out["p95_ms"] < p95_limit_ms),
        }
    return out


def estimate_drops_from_period_ns(stamps_ns: Iterable[int], expected_fps: float) -> dict[str, Any]:
    stamps = [int(s) for s in stamps_ns if s is not None]
    stamps.sort()
    if expected_fps <= 0:
        raise ValueError("expected_fps must be > 0")
    if len(stamps) < 2:
        return {"drop_estimate_frames": 0, "drop_rate_estimate": 0.0}

    period_ns = 1e9 / expected_fps
    missing = 0
    for a, b in zip(stamps[:-1], stamps[1:]):
        dt = max(0.0, float(b - a))
        if dt < 1.5 * period_ns:
            continue
        missing += max(0, int(round(dt / period_ns)) - 1)

    valid_frames = len(stamps)
    return {
        "drop_estimate_frames": int(missing),
        "drop_rate_estimate": float(missing / max(1, valid_frames + missing)),
    }


def summarize_image_health(
    recv_stamps_ns: Iterable[int],
    header_stamps_ns: Iterable[int] | None,
    expected_fps: float,
    latency_p95_limit_ms: float | None = None,
) -> dict[str, Any]:
    recv = [int(s) for s in recv_stamps_ns if s is not None]
    recv.sort()
    header = [int(s) for s in (header_stamps_ns or []) if s is not None]
    header.sort()

    duration_sec = 0.0
    if len(recv) >= 2:
        duration_sec = (recv[-1] - recv[0]) / 1e9
    fps = float(len(recv) / duration_sec) if duration_sec > 0 else (float(len(recv)) if recv else 0.0)

    lat_ms: list[float] = []
    if header_stamps_ns is not None:
        for r, h in zip(recv_stamps_ns, header_stamps_ns):
            if r is None or h is None:
                continue
            lat_ms.append((int(r) - int(h)) / 1e6)

    out = {
        "frames": len(recv),
        "duration_sec": float(duration_sec),
        "fps": fps,
        "expected_fps": float(expected_fps),
        "fps_ratio": float(fps / expected_fps) if expected_fps > 0 else None,
        "drop": estimate_drops_from_period_ns(header if header else recv, expected_fps),
        "latency": summarize_latency_ms(lat_ms, p95_limit_ms=latency_p95_limit_ms) if lat_ms else summarize_latency_ms([], p95_limit_ms=latency_p95_limit_ms),
    }
    return out


def greedy_approx_sync_pairs_ns(
    left_stamps_ns: Iterable[int],
    right_stamps_ns: Iterable[int],
    slop_ms: float,
) -> dict[str, Any]:
    left = sorted(int(x) for x in left_stamps_ns if x is not None)
    right = sorted(int(x) for x in right_stamps_ns if x is not None)
    slop_ns = int(slop_ms * 1e6)
    i = 0
    j = 0
    pairs = 0
    deltas_ms: list[float] = []
    while i < len(left) and j < len(right):
        d = left[i] - right[j]
        ad = abs(d)
        if ad <= slop_ns:
            pairs += 1
            deltas_ms.append(ad / 1e6)
            i += 1
            j += 1
            continue
        if d < 0:
            i += 1
        else:
            j += 1

    denom = min(len(left), len(right))
    success_rate = float(pairs / denom) if denom > 0 else 0.0
    return {
        "left_count": len(left),
        "right_count": len(right),
        "pairs": pairs,
        "slop_ms": float(slop_ms),
        "success_rate": success_rate,
        "pair_abs_delta_ms": summarize_latency_ms(deltas_ms),
    }


def summarize_pose_jitter(points_xyz: Iterable[Iterable[float]], std_limit_m: float = 0.003) -> dict[str, Any]:
    pts = np.asarray(list(points_xyz), dtype=float)
    if pts.size == 0:
        return {
            "count": 0,
            "mean_xyz_m": None,
            "std_xyz_m": None,
            "radial_std_m": None,
            "gate": {"std_limit_m": float(std_limit_m), "pass": False, "reason": "no_samples"},
        }
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be Nx3")

    mean_xyz = np.mean(pts, axis=0)
    std_xyz = np.std(pts, axis=0)
    radial = np.linalg.norm(pts - mean_xyz, axis=1)
    radial_std = float(np.std(radial))
    gate_pass = bool(np.all(std_xyz < std_limit_m))
    return {
        "count": int(pts.shape[0]),
        "mean_xyz_m": [float(v) for v in mean_xyz],
        "std_xyz_m": [float(v) for v in std_xyz],
        "radial_std_m": radial_std,
        "gate": {
            "std_limit_m": float(std_limit_m),
            "pass": gate_pass,
            "axes": {
                "x": bool(std_xyz[0] < std_limit_m),
                "y": bool(std_xyz[1] < std_limit_m),
                "z": bool(std_xyz[2] < std_limit_m),
            },
        },
    }


def summarize_id_switch(ids: Iterable[Any], valid_flags: Iterable[bool] | None = None, missing_warn_rate: float = 0.05) -> dict[str, Any]:
    id_list = list(ids)
    flags = list(valid_flags) if valid_flags is not None else [True] * len(id_list)
    if len(flags) != len(id_list):
        raise ValueError("valid_flags length must match ids length")

    total = len(id_list)
    missing = 0
    valid_ids: list[Any] = []
    for obj_id, is_valid in zip(id_list, flags):
        if not is_valid or obj_id is None or obj_id == "":
            missing += 1
            continue
        valid_ids.append(obj_id)

    switch_events = 0
    for prev, cur in zip(valid_ids[:-1], valid_ids[1:]):
        if cur != prev:
            switch_events += 1

    valid_frames = len(valid_ids)
    switch_rate = float(switch_events / valid_frames) if valid_frames > 0 else None
    missing_rate = float(missing / total) if total > 0 else None
    warnings: list[str] = []
    if missing_rate is not None and missing_rate >= missing_warn_rate:
        warnings.append(f"missing_rate >= {missing_warn_rate:.3f}")

    return {
        "total_frames": total,
        "valid_frames": valid_frames,
        "missing_frames": missing,
        "switch_events": switch_events,
        "switch_rate": switch_rate,
        "missing_rate": missing_rate,
        "warnings": warnings,
    }


def summarize_state_topic_latency_by_topic(
    topic_to_latencies_ms: dict[str, Iterable[float]],
    p95_limit_ms: float,
) -> dict[str, Any]:
    per_topic: dict[str, Any] = {}
    all_vals: list[float] = []
    for topic, vals in topic_to_latencies_ms.items():
        vals_clean = _clean_floats(vals)
        all_vals.extend(vals_clean)
        per_topic[topic] = summarize_latency_ms(vals_clean, p95_limit_ms=p95_limit_ms)
    overall = summarize_latency_ms(all_vals, p95_limit_ms=p95_limit_ms)
    return {
        "overall": overall,
        "per_topic": per_topic,
        "gate_basis": "state_topics_only",
    }
