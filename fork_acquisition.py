#!/usr/bin/env python3
"""
fork_acquisition.py

Vertical Skewer (VS) food acquisition for the Kinova j2s6s200 + RealSense setup.

Given a bounding box [u_min, v_min, u_max, v_max] in pixel space:
  1. Sample median depth in a 5x5 patch at the bbox centroid → 3D skewer target
  2. Phase 1 — Y-align:  J2 (elbow pitch)
  3. Phase 2 — X-align:  J3 (forearm rotation)
  4. Phase 3 — Hover:    J1+J4, stop HOVER_HEIGHT above food surface
  5. Phase 4 — Plunge:   J1+J4, drive PENETRATE_DEPTH past food surface (fast)
  6. Phase 5 — Retract:  J1+J4, pull back to hover height

Trigger via ROS2 topic:
  ros2 topic pub /fork_acquisition/bbox std_msgs/Int32MultiArray \
    "data: [u_min, v_min, u_max, v_max]"

Or call node.acquire(u_min, v_min, u_max, v_max) from another node.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32MultiArray
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── Camera intrinsics ─────────────────────────────────────────────────────────
FX = 603.6312;  FY = 603.0632
CX = 319.0870;  CY = 236.3678

# Camera is mounted above the EE tip — empirical offsets compensate for this.
# Camera optical Y points downward, so subtracting CAM_Y_OFFSET shifts aim down to EE level.
CAM_Y_OFFSET =  0.09    # m — camera sits above EE along camera-Y axis
CAM_X_OFFSET = -0.187   # m — lateral camera offset from EE centreline

# ── Frames ────────────────────────────────────────────────────────────────────
BASE_FRAME = "j2s6s200_link_base"
CAM_FRAME  = "camera_color_optical_frame"
EE_LINK    = "j2s6s200_end_effector"

ARM_JOINT_NAMES = [
    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
]

# ── Acquisition geometry ──────────────────────────────────────────────────────
HOVER_HEIGHT    = 0.06   # m above food surface before plunge
PENETRATE_DEPTH = 0.02   # m below food surface for tine grip
DEPTH_PATCH_PX  = 5      # half-side of depth-sampling square (pixels)

# ── Joint-space control (same empirical tuning as click_pointer.py) ───────────
STEP_RADS    = 0.05   # joint step magnitude per phase
CART_DELTA_X = 0.15   # m EE X travel per STEP_RADS on J3
CART_DELTA_Y = 0.12   # m EE Y travel per STEP_RADS on J2
CART_DELTA_Z = 0.06   # m EE Z travel per STEP_RADS on J1/J4 combined
TICK_DUR_S   = 0.6    # seconds per proportional step
ERR_TOL_M    = 0.04   # skip phase if error is below this threshold


class ForkAcquirer(Node):
    def __init__(self):
        super().__init__("fork_acquirer")
        self._cb    = ReentrantCallbackGroup()
        self._lock  = threading.Lock()
        self.bridge = CvBridge()

        self._depth_img: np.ndarray | None = None
        self._joints = {n: 0.0 for n in ARM_JOINT_NAMES}
        self.busy    = False
        self.phase   = ""

        self._tf = Buffer()
        TransformListener(self._tf, self)

        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self._depth_cb, 10, callback_group=self._cb)
        self.create_subscription(
            JointState, "/joint_states",
            self._js_cb, 10, callback_group=self._cb)
        self.create_subscription(
            Int32MultiArray, "/fork_acquisition/bbox",
            self._bbox_cb, 10, callback_group=self._cb)

        self._traj = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
            callback_group=self._cb)
        self.get_logger().info("Waiting for trajectory controller...")
        if not self._traj.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("Trajectory controller not available")

        self.get_logger().info(
            "ForkAcquirer ready.\n"
            "  Trigger: ros2 topic pub /fork_acquisition/bbox "
            "std_msgs/Int32MultiArray \"data: [u_min,v_min,u_max,v_max]\""
        )

    # ── Subscribers ───────────────────────────────────────────────────────────

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def _js_cb(self, msg):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._joints:
                    self._joints[name] = pos

    def _bbox_cb(self, msg):
        if len(msg.data) != 4:
            self.get_logger().warn(
                f"bbox topic expects [u_min,v_min,u_max,v_max], got {len(msg.data)} values")
            return
        self.acquire(*msg.data)

    # ── Public entry point ────────────────────────────────────────────────────

    def acquire(self, u_min: int, v_min: int, u_max: int, v_max: int):
        if self.busy:
            self.get_logger().warn("Arm busy — ignoring new bbox")
            return
        with self._lock:
            depth = self._depth_img
        if depth is None:
            self.get_logger().warn("No depth image yet")
            return

        target_cam = self._bbox_to_3d(int(u_min), int(v_min), int(u_max), int(v_max), depth)
        if target_cam is None:
            return

        threading.Thread(
            target=self._run_acquisition, args=(target_cam,), daemon=True).start()

    # ── Bbox → 3-D target ────────────────────────────────────────────────────

    def _bbox_to_3d(self, u_min, v_min, u_max, v_max,
                    depth: np.ndarray) -> np.ndarray | None:
        u_c = (u_min + u_max) // 2
        v_c = (v_min + v_max) // 2

        h, w = depth.shape[:2]
        r    = DEPTH_PATCH_PX

        # Sample window: 5x5 centred on bbox centroid, clamped to bbox interior.
        # Depth pixels at bbox edges are mixed (foreground/background) and noisy —
        # staying inside the bbox keeps samples on the food surface.
        u0 = max(u_min, u_c - r, 0)
        u1 = min(u_max, u_c + r, w - 1)
        v0 = max(v_min, v_c - r, 0)
        v1 = min(v_max, v_c + r, h - 1)

        patch = depth[v0:v1 + 1, u0:u1 + 1].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size == 0:
            self.get_logger().error(
                f"All depth values in patch [{u0}:{u1},{v0}:{v1}] are zero")
            return None

        z_m = float(np.median(valid)) * 0.001  # uint16 mm → float m
        if z_m < 0.05:
            self.get_logger().error(
                f"Depth {z_m*1000:.0f} mm too small — occluded or sensor error")
            return None

        x_m = (u_c - CX) * z_m / FX + CAM_X_OFFSET
        y_m = (v_c - CY) * z_m / FY - CAM_Y_OFFSET

        self.get_logger().info(
            f"Bbox ({u_min},{v_min})→({u_max},{v_max})  centroid ({u_c},{v_c})"
            f"  depth={z_m*100:.1f} cm"
            f"  target_cam=[{x_m:+.3f}, {y_m:+.3f}, {z_m:+.3f}]"
        )
        return np.array([x_m, y_m, z_m])

    # ── Acquisition sequence ──────────────────────────────────────────────────

    def _run_acquisition(self, target_cam: np.ndarray):
        self.busy = True
        try:
            # Snapshot EE position in camera frame before XY alignment.
            try:
                tf_ee = self._tf.lookup_transform(
                    CAM_FRAME, EE_LINK, Time(), Duration(seconds=1.0))
            except Exception as e:
                self.get_logger().error(f"TF EE→cam failed: {e}")
                return

            ee = np.array([
                tf_ee.transform.translation.x,
                tf_ee.transform.translation.y,
                tf_ee.transform.translation.z,
            ])
            self.get_logger().info(
                f"EE start (cam): [{ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}]")

            # ── Phase 1: Y-align (J2, elbow pitch) ───────────────────────────
            dy = target_cam[1] - ee[1]
            self.phase = "1/5 Y-align"
            self.get_logger().info(f"Phase 1 — Y-align  dy={dy:+.3f} m")
            if abs(dy) > ERR_TOL_M:
                delta = -(dy / CART_DELTA_Y) * STEP_RADS
                dur   = max(1.5, abs(dy) / CART_DELTA_Y * TICK_DUR_S)
                self._step({1: delta}, dur)

            # ── Phase 2: X-align (J3, forearm rotation) ───────────────────────
            # target_cam[0] is the signed lateral offset from the camera optical
            # axis — zero when the bbox centroid is image-centred.
            dx = target_cam[0]
            self.phase = "2/5 X-align"
            self.get_logger().info(f"Phase 2 — X-align  dx={dx:+.3f} m")
            if abs(dx) > ERR_TOL_M:
                delta = -(dx / CART_DELTA_X) * STEP_RADS
                dur   = max(1.5, abs(dx) / CART_DELTA_X * TICK_DUR_S)
                self._step({2: delta}, dur)

            # Re-read EE Z after XY alignment because J2/J3 motion shifts the
            # arm forward and changes the camera-frame depth offset.
            try:
                tf_ee2 = self._tf.lookup_transform(
                    CAM_FRAME, EE_LINK, Time(), Duration(seconds=1.0))
                ee_z_now = tf_ee2.transform.translation.z
            except Exception:
                ee_z_now = ee[2]

            # ── Phase 3: Hover (J1+J4, stop HOVER_HEIGHT above food) ──────────
            z_hover   = target_cam[2] - HOVER_HEIGHT
            dz_hover  = z_hover - ee_z_now
            self.phase = "3/5 Hover"
            self.get_logger().info(
                f"Phase 3 — Hover  z_food={target_cam[2]:.3f}  "
                f"z_hover={z_hover:.3f}  dz={dz_hover:+.3f}")
            if dz_hover > ERR_TOL_M:
                delta = (dz_hover / CART_DELTA_Z) * STEP_RADS * 2.0
                dur   = max(1.5, dz_hover / CART_DELTA_Z * TICK_DUR_S)
                self._step({0: delta, 3: delta}, dur)

            # ── Phase 4: Plunge (fast — food cannot escape during descent) ────
            dz_plunge = HOVER_HEIGHT + PENETRATE_DEPTH
            plunge_j  = (dz_plunge / CART_DELTA_Z) * STEP_RADS * 2.0
            self.phase = "4/5 Plunge"
            self.get_logger().info(f"Phase 4 — Plunge  dz={dz_plunge:.3f} m (fast)")
            dur_plunge = max(0.8, dz_plunge / CART_DELTA_Z * TICK_DUR_S * 0.5)
            self._step({0: plunge_j, 3: plunge_j}, dur_plunge)

            time.sleep(0.3)  # dwell at contact before pulling back

            # ── Phase 5: Retract ──────────────────────────────────────────────
            self.phase = "5/5 Retract"
            self.get_logger().info(f"Phase 5 — Retract  dz={-dz_plunge:.3f} m")
            dur_ret = max(1.0, dz_plunge / CART_DELTA_Z * TICK_DUR_S)
            self._step({0: -plunge_j, 3: -plunge_j}, dur_ret)

            self.get_logger().info("Fork acquisition complete.")

        except Exception as e:
            self.get_logger().error(f"Acquisition failed: {e}")
        finally:
            self.busy = False
            self.phase = ""

    # ── Trajectory dispatch ───────────────────────────────────────────────────

    def _step(self, deltas: dict, duration_s: float):
        """Build and send one joint trajectory waypoint; blocks until done."""
        with self._lock:
            cur = [self._joints[n] for n in ARM_JOINT_NAMES]

        tgt = list(cur)
        for idx, d in deltas.items():
            tgt[idx] += d

        pt0 = JointTrajectoryPoint()
        pt0.positions       = cur
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

        dur_s = int(duration_s)
        pt1 = JointTrajectoryPoint()
        pt1.positions       = tgt
        pt1.time_from_start = DurationMsg(
            sec=dur_s, nanosec=int((duration_s - dur_s) * 1e9))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt0, pt1]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        done = threading.Event()

        def _on_goal(f):
            gh = f.result()
            if gh.accepted:
                gh.get_result_async().add_done_callback(lambda _: done.set())
            else:
                self.get_logger().warn("Trajectory goal rejected")
                done.set()

        self._traj.send_goal_async(goal).add_done_callback(_on_goal)
        done.wait()


# ── Standalone entry point ────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ForkAcquirer()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
