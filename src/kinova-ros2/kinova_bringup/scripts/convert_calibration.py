#!/usr/bin/env python3
"""
Convert calibration result from EE→camera_color_optical_frame
to EE→camera_link (what sample.launch.py expects).

Usage:
  python3 convert_calibration.py

Input: EE→camera_color_optical_frame (from MoveIt handeye calibration panel)
Output: EE→camera_link values to paste into sample.launch.py
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ── Paste your calibration result here ──────────────────────────────────────
# EE -> camera_color_optical_frame  (from MoveIt handeye panel, after inverting
# the "sensor->EE" output it shows)
EE_TO_OPT_translation = np.array([0.0, 0.0, 0.0])   # ← replace with your values (metres)
EE_TO_OPT_quat_xyzw   = np.array([0.0, 0.0, 0.0, 1.0])  # ← replace with your quaternion

# ── camera_link → camera_color_optical_frame  (read from your live system) ──
# These come from the RealSense driver and are hardware-specific.
# Run: ros2 run tf2_ros tf2_echo camera_link camera_color_optical_frame
CAM_LINK_TO_OPT_translation = np.array([-0.000, 0.015, 0.000])
CAM_LINK_TO_OPT_quat_xyzw   = np.array([-0.497, 0.503, -0.498, 0.502])

# ── Compute EE → camera_link ─────────────────────────────────────────────────
R_ee_opt  = Rotation.from_quat(EE_TO_OPT_quat_xyzw)
R_cl_opt  = Rotation.from_quat(CAM_LINK_TO_OPT_quat_xyzw)
t_cl_opt  = CAM_LINK_TO_OPT_translation

# T(EE→camera_link) = T(EE→opt) * T(camera_link→opt)^-1
R_opt_cl  = R_cl_opt.inv()                                  # camera_color_optical → camera_link
t_opt_cl  = R_opt_cl.apply(-t_cl_opt)                       # translation of camera_link in optical frame

R_ee_cl   = R_ee_opt * R_opt_cl
t_ee_cl   = EE_TO_OPT_translation + R_ee_opt.apply(t_opt_cl)

rpy = R_ee_cl.as_euler('xyz', degrees=False)
q   = R_ee_cl.as_quat()

print("=" * 60)
print("EE → camera_link  (paste into sample.launch.py)")
print("=" * 60)
print(f"  cam_x:     {t_ee_cl[0]:.5f}")
print(f"  cam_y:     {t_ee_cl[1]:.5f}")
print(f"  cam_z:     {t_ee_cl[2]:.5f}")
print(f"  cam_roll:  {rpy[0]:.5f}  ({np.degrees(rpy[0]):.2f}°)")
print(f"  cam_pitch: {rpy[1]:.5f}  ({np.degrees(rpy[1]):.2f}°)")
print(f"  cam_yaw:   {rpy[2]:.5f}  ({np.degrees(rpy[2]):.2f}°)")
print(f"\n  (quaternion xyzw: {q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f})")
print()

# Sanity check: compose and print total EE→camera_color_optical_frame
R_check = R_ee_cl * R_cl_opt
t_check = t_ee_cl + R_ee_cl.apply(t_cl_opt)
rpy_check = R_check.as_euler('xyz', degrees=True)
print("Sanity — total EE → camera_color_optical_frame should match calibration input:")
print(f"  translation: {t_check}")
print(f"  RPY (deg):   {rpy_check}")
