from __future__ import annotations

import json
import numpy as np

from hrl_trainer.v5_1.ee_fk import ee_pose6_from_q


def main() -> None:
    q_base = np.array([0.08, -0.35, 0.62, -1.05, 1.12, -0.48, 0.31], dtype=float)
    q_rack_shifted = q_base.copy()
    q_rack_shifted[0] += 0.15  # only rack changes

    # Before fix behavior (rack locked): overwrite shifted rack back to base rack.
    ee_pose_before = ee_pose6_from_q(np.array([q_base[0], *q_base[1:]], dtype=float))
    ee_pose_before_after_shift = ee_pose6_from_q(np.array([q_base[0], *q_rack_shifted[1:]], dtype=float))
    delta_before = ee_pose_before_after_shift - ee_pose_before

    # After fix behavior (rack is a normal controlled joint).
    ee_pose_after = ee_pose6_from_q(q_rack_shifted)
    delta_after = ee_pose_after - ee_pose_before

    payload = {
        "q_base": q_base.tolist(),
        "q_rack_shifted": q_rack_shifted.tolist(),
        "ee_pose_before": ee_pose_before.tolist(),
        "ee_pose_after": ee_pose_after.tolist(),
        "delta": delta_after.tolist(),
        "delta_before_fix_locked_rack": delta_before.tolist(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
