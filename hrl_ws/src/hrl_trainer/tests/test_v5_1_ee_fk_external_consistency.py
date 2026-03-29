from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from hrl_trainer.v5_1.ee_fk import ee_pose6_from_q


class TestV51EEFKExternalConsistency(unittest.TestCase):
    def test_fixed_q_matches_external_controller_fk(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        external_file = (
            repo_root
            / "external"
            / "ENPM662_Group4_FinalProject"
            / "src"
            / "kitchen_robot_controller"
            / "kitchen_robot_controller"
            / "kinematics.py"
        )

        spec = importlib.util.spec_from_file_location("external_kinematics", external_file)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        q7 = np.array([0.12, -0.35, 0.48, -0.62, 0.27, -0.14, 0.51], dtype=float)
        T_ext = module.fk_ur(q7)
        R = T_ext[:3, :3]
        ext_pose6 = np.array(
            [
                T_ext[0, 3],
                T_ext[1, 3],
                T_ext[2, 3],
                np.arctan2(R[2, 1], R[2, 2]),
                np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)),
                np.arctan2(R[1, 0], R[0, 0]),
            ],
            dtype=float,
        )

        v51_pose6 = ee_pose6_from_q(q7)
        self.assertTrue(np.allclose(v51_pose6, ext_pose6, atol=1e-9, rtol=0.0), msg=f"v5_1={v51_pose6}, ext={ext_pose6}")


if __name__ == "__main__":
    unittest.main()
