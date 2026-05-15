#!/usr/bin/env python3
"""
calibrate_deltas.py — empirically measure CART_DELTA_X/Y/Z for fork_acquisition.py

How to use
----------
1. Start food_segmenter so /fork_acquisition/bbox is published.
2. Run this script.  The arm must be in gravity-compensation / teach-pendant mode
   so you can move it by hand.
3. With food visible, press Enter → snapshots START joints + camera-frame target.
4. Manually push the arm so the fork tip sits at the food.
5. Press Enter again → snapshots END joints, prints recommended CART_DELTA values.

Maths
-----
Control law in fork_acquisition:
    delta_rad = (error_m / CART_DELTA) * STEP_RADS
Rearranged to solve for CART_DELTA:
    CART_DELTA = error_m * STEP_RADS / delta_rad

So for each axis we need:
  - error_m   : initial camera-frame distance (from bbox + depth at START)
  - delta_rad : how many radians the responsible joint actually moved (END - START)
"""
from __future__ import annotations

import os
import threading

os.environ["QT_LOGGING_RULES"] = "qt.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32MultiArray
from tf2_ros import Buffer, TransformListener

FX, FY = 603.6312, 603.0632
CX, CY = 319.0870, 236.3678

CAM_FRAME  = "camera_color_optical_frame"
BASE_FRAME = "j2s6s200_link_base"
EE_LINK    = "j2s6s200_end_effector"

ARM_JOINT_NAMES = [
    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
]

# Current offsets — same as fork_acquisition.py
CAM_X_OFFSET = -0.15
CAM_Y_OFFSET =  0.00
CAM_Z_OFFSET = -0.15
DEPTH_PATCH  =  4
STEP_RADS    =  0.05


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y],
    ])


class CalibratorNode(Node):

    def __init__(self):
        super().__init__("calibrate_deltas")
        self._lock     = threading.Lock()
        self._bridge   = CvBridge()
        self._joints   = {n: 0.0 for n in ARM_JOINT_NAMES}
        self._depth    = None
        self._bbox     = None

        self._tf = Buffer()
        TransformListener(self._tf, self)

        self.create_subscription(JointState, "/joint_states",          self._js_cb,    10)
        self.create_subscription(Image,      "/camera/camera/aligned_depth_to_color/image_raw",
                                             self._depth_cb, 10)
        self.create_subscription(Int32MultiArray, "/fork_acquisition/bbox",
                                             self._bbox_cb,  10)

        self.get_logger().info("Subscribed — waiting for joint states, depth, and bbox ...")

    def _js_cb(self, msg):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._joints:
                    self._joints[name] = pos

    def _depth_cb(self, msg):
        with self._lock:
            self._depth = self._bridge.imgmsg_to_cv2(msg, "passthrough")

    def _bbox_cb(self, msg):
        if len(msg.data) == 4:
            with self._lock:
                self._bbox = tuple(msg.data)

    # ── snapshot helpers ──────────────────────────────────────────────────────

    def snapshot_joints(self) -> dict:
        with self._lock:
            return dict(self._joints)

    def snapshot_target_cam(self):
        """
        Returns (target_cam, ee_cam, errors) or None on failure.
        target_cam : 3-D point in camera frame at the food (with EE offsets)
        ee_cam     : EE position in camera frame right now
        errors     : (dx, dy, dz) = target_cam - ee_cam
        """
        with self._lock:
            bbox  = self._bbox
            depth = self._depth.copy() if self._depth is not None else None

        if bbox is None:
            print("[ERR] no bbox yet — is food_segmenter running?")
            return None
        if depth is None:
            print("[ERR] no depth image yet")
            return None

        x1, y1, x2, y2 = bbox
        u, v = (x1 + x2) // 2, (y1 + y2) // 2

        h, w = depth.shape[:2]
        r  = DEPTH_PATCH
        y0 = max(0, v - r);  y1e = min(h, v + r + 1)
        x0 = max(0, u - r);  x1e = min(w, u + r + 1)
        patch = depth[y0:y1e, x0:x1e].astype(np.float32)
        valid = patch[patch > 10]
        if valid.size < 4:
            print(f"[ERR] no valid depth at ({u},{v})")
            return None

        z_raw = float(np.median(valid)) * 0.001
        if not (0.05 < z_raw < 1.5):
            print(f"[ERR] depth {z_raw*100:.1f} cm out of range")
            return None

        x_cam = (u - CX) * z_raw / FX
        y_cam = (v - CY) * z_raw / FY
        target_cam = np.array([x_cam + CAM_X_OFFSET,
                                y_cam - CAM_Y_OFFSET,
                                z_raw + CAM_Z_OFFSET])

        try:
            tf = self._tf.lookup_transform(CAM_FRAME, EE_LINK, Time(), Duration(seconds=1.0))
            ee_cam = np.array([tf.transform.translation.x,
                               tf.transform.translation.y,
                               tf.transform.translation.z])
        except Exception as e:
            print(f"[ERR] TF EE→cam failed: {e}")
            return None

        dx = target_cam[0] - ee_cam[0]   # X error in cam frame (→ J1)
        dy = target_cam[1] - ee_cam[1]   # Y error in cam frame (→ J3)
        dz = target_cam[2] - ee_cam[2]   # Z error in cam frame (→ J1+J4 hover, J2+J3 plunge)

        return target_cam, ee_cam, (dx, dy, dz), (u, v), z_raw


def _fmt_joints(j: dict) -> str:
    return "  ".join(f"J{i+1}={j[n]:+.4f}" for i, n in enumerate(ARM_JOINT_NAMES))


def main():
    rclpy.init()
    node = CalibratorNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    print("\n" + "=" * 60)
    print("  CART_DELTA Calibration Tool")
    print("=" * 60)
    print("  Joint → axis mapping in fork_acquisition.py:")
    print("    J1 (index 0) → X-align")
    print("    J3 (index 2) → Y-align")
    print("    J1+J4        → Z-hover approach")
    print("    J2+J3        → Plunge / retract")
    print("=" * 60)
    print()

    input("  [Step 1]  Arm at rest, food visible — press Enter to snapshot START ...")

    snap = node.snapshot_target_cam()
    if snap is None:
        print("[ABORT] could not get start snapshot.")
        rclpy.shutdown(); return

    target_cam, ee_cam, (dx, dy, dz), (u, v), z_raw = snap
    j_start = node.snapshot_joints()

    print()
    print(f"  Bbox centre    : pixel ({u}, {v})")
    print(f"  Depth          : {z_raw*100:.1f} cm")
    print(f"  Target cam     : [{target_cam[0]:+.4f}, {target_cam[1]:+.4f}, {target_cam[2]:+.4f}] m")
    print(f"  EE cam         : [{ee_cam[0]:+.4f}, {ee_cam[1]:+.4f}, {ee_cam[2]:+.4f}] m")
    print()
    print(f"  Initial errors :")
    print(f"    dx (→ J1)    : {dx:+.4f} m")
    print(f"    dy (→ J3)    : {dy:+.4f} m")
    print(f"    dz (hover)   : {dz:+.4f} m")
    print()
    print(f"  Start joints   : {_fmt_joints(j_start)}")
    print()

    input("  [Step 2]  Manually move arm so fork tip is AT the food — press Enter to snapshot END ...")

    j_end = node.snapshot_joints()
    print()
    print(f"  End joints     : {_fmt_joints(j_end)}")

    deltas = {n: j_end[n] - j_start[n] for n in ARM_JOINT_NAMES}
    print()
    print("  Joint deltas (end - start):")
    for i, n in enumerate(ARM_JOINT_NAMES):
        print(f"    J{i+1}: {deltas[n]:+.4f} rad")

    print()
    print("=" * 60)
    print("  Recommended CART_DELTA values")
    print("  Formula: CART_DELTA = |error_m| * STEP_RADS / |delta_rad|")
    print("=" * 60)

    dj1 = deltas[ARM_JOINT_NAMES[0]]   # J1
    dj2 = deltas[ARM_JOINT_NAMES[1]]   # J2
    dj3 = deltas[ARM_JOINT_NAMES[2]]   # J3
    dj4 = deltas[ARM_JOINT_NAMES[3]]   # J4

    if abs(dj1) > 1e-4 and abs(dx) > 0.01:
        cdx = abs(dx) * STEP_RADS / abs(dj1)
        print(f"  CART_DELTA_X = {cdx:.4f}  (from dx={dx:+.4f} m, ΔJ1={dj1:+.4f} rad)")
    else:
        print(f"  CART_DELTA_X : skipped — dx={dx:+.4f} m or ΔJ1={dj1:+.4f} rad too small")

    if abs(dj3) > 1e-4 and abs(dy) > 0.01:
        cdy = abs(dy) * STEP_RADS / abs(dj3)
        print(f"  CART_DELTA_Y = {cdy:.4f}  (from dy={dy:+.4f} m, ΔJ3={dj3:+.4f} rad)")
    else:
        print(f"  CART_DELTA_Y : skipped — dy={dy:+.4f} m or ΔJ3={dj3:+.4f} rad too small")

    # Z-hover uses J1+J4 both with same delta; formula uses average
    dj14_avg = (abs(dj1) + abs(dj4)) / 2.0
    if dj14_avg > 1e-4 and abs(dz) > 0.01:
        cdz = abs(dz) * STEP_RADS / (dj14_avg * 2.0)
        print(f"  CART_DELTA_Z = {cdz:.4f}  (from dz={dz:+.4f} m, avg(|ΔJ1|,|ΔJ4|)={dj14_avg:.4f} rad)")
    else:
        print(f"  CART_DELTA_Z : skipped — dz={dz:+.4f} m or J1+J4 delta too small")

    # Plunge uses J2+J3 — compute what dz that corresponds to
    dj23_avg = (abs(dj2) + abs(dj3)) / 2.0
    if dj23_avg > 1e-4:
        cdz_plunge = abs(dz) * STEP_RADS / (dj23_avg * 2.0)
        print(f"  CART_DELTA_Z (plunge J2+J3) = {cdz_plunge:.4f}  "
              f"(avg(|ΔJ2|,|ΔJ3|)={dj23_avg:.4f} rad)")

    print()
    print("  Copy the above values into fork_acquisition.py constants.")
    print("=" * 60)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
