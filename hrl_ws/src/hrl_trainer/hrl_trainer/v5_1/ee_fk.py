"""External-controller-consistent EE FK for V5.1.

This module mirrors the matrix-based FK algorithm used by
`external/.../kitchen_robot_controller/kinematics.py::fk_ur`.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_JOINT_TYPES: tuple[str, ...] = (
    "prismatic",
    "revolute",
    "revolute",
    "revolute",
    "revolute",
    "continuous",
    "revolute",
)

_ORIGIN_XYZ = np.array(
    [
        [0.00715921043213119, 0.0000809621375843506, -0.0635],
        [-0.021178, 0.0, 0.1868],
        [-0.0633967414837172, 0.000642782425827271, 0.0602000000000009],
        [-0.000134989688424625, 0.425, 0.0133123982251372],
        [-0.0000850456535865796, -0.39225, -0.0083864861805065],
        [0.0475482889721905, -0.000817137634885778, -0.0805958577476871],
        [0.0436977540622506, 0.000443046177049933, -0.0521517110277254],
    ],
    dtype=float,
)

_ORIGIN_RPY = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.5707963267949, 0.0, 1.5707963267949],
        [3.14159265358979, 0.0, 0.0],
        [3.14159265358979, 0.0, -1.5707963267949],
        [3.14159265358979, 1.5707963267949, 0.0],
        [-1.5707963267949, 0.0, -1.5707963267949],
    ],
    dtype=float,
)

_AXES_LOCAL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0101382310641698, 0.0, -0.999948606814815],
        [0.010138231064165, 0.0, 0.999948606814815],
        [0.0, -0.0101382310641647, -0.999948606814815],
        [0.0, 0.0, -1.0],
        [-0.0101384515502096, 0.0, 0.999948604579338],
    ],
    dtype=float,
)


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _rot_axis_local(axis_local: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis_local, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def _make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def fk_matrix_from_q7(q: Sequence[float]) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    if q_arr.shape[0] != 7:
        raise ValueError("Expected 7 joint values [q_rack, q1..q6]")

    T_W = np.eye(4, dtype=float)
    for i in range(7):
        ox, oy, oz = _ORIGIN_XYZ[i]
        rr, pp, yy = _ORIGIN_RPY[i]
        T_W = T_W @ _make_T(_rpy_to_rot(rr, pp, yy), np.array([ox, oy, oz], dtype=float))

        qi = q_arr[i]
        if _JOINT_TYPES[i] in ("revolute", "continuous"):
            T_motion = _make_T(_rot_axis_local(_AXES_LOCAL[i], qi), np.zeros(3, dtype=float))
        elif _JOINT_TYPES[i] == "prismatic":
            T_motion = _make_T(np.eye(3, dtype=float), _AXES_LOCAL[i] * qi)
        else:
            raise ValueError(f"Unsupported joint type '{_JOINT_TYPES[i]}'")
        T_W = T_W @ T_motion
    return T_W


def ee_pose6_from_q(q: Sequence[float]) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    if q_arr.shape[0] == 6:
        q7 = np.concatenate([np.array([0.0], dtype=float), q_arr], axis=0)
    elif q_arr.shape[0] == 7:
        q7 = q_arr
    else:
        raise ValueError("Expected q length 6 (q1..q6) or 7 (q_rack,q1..q6)")

    T = fk_matrix_from_q7(q7)
    R = T[:3, :3]
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw], dtype=float)
