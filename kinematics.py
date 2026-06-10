"""Exact forward kinematics for the Kinova j2s6s200, transcribed from the URDF.

Every transform comes from j2s6s200.xacro (joint origins + EE offset). The chain
is the standard URDF convention: link_i = link_(i-1) @ Origin_i @ Rz(q_i).

This module is pure numpy so it is shared verbatim by the sim env and (optionally)
the hardware node for sanity checks. No ROS / torch dependency.
"""
from __future__ import annotations

import numpy as np


def _rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF fixed-axis XYZ == Rz @ Ry @ Rx


def _xform(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(rpy)
    T[:3, 3] = xyz
    return T


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


class KinovaFK:
    """Forward kinematics + wrist-camera pose for the j2s6s200."""

    def __init__(self, cfg: dict):
        chain = cfg["arm"]["chain"]
        self._origins = [_xform(j["xyz"], j["rpy"]) for j in chain]
        self._ee = _xform(cfg["arm"]["ee_offset"]["xyz"], cfg["arm"]["ee_offset"]["rpy"])
        cam = cfg["camera"]["ee_to_cam"]
        self._ee_to_cam = _xform(cam["xyz"], cam["rpy"])
        self.n = len(self._origins)

    def fk_ee(self, q: np.ndarray) -> np.ndarray:
        """4x4 base->end_effector transform for joint angles q (rad)."""
        T = np.eye(4)
        for i in range(self.n):
            T = T @ self._origins[i] @ _rotz(q[i])
        return T @ self._ee

    def fk_cam(self, q: np.ndarray) -> np.ndarray:
        """4x4 base->camera_optical_frame transform."""
        return self.fk_ee(q) @ self._ee_to_cam

    def ee_pose(self, q: np.ndarray) -> np.ndarray:
        """End-effector pose as [x, y, z, qx, qy, qz, qw]."""
        T = self.fk_ee(q)
        return np.concatenate([T[:3, 3], R_to_quat(T[:3, :3])])
