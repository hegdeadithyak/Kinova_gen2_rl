#!/usr/bin/env python3
"""
goto_pixel.py — Move EE to a clicked pixel in the depth camera frame via MoveIt IK.

Architecture:
    pixel (u,v) + depth → pinhole backproject → camera-frame 3D point
    → TF camera→base → base-frame 3D point
    → MoveIt /compute_ik → joint solution
    → /arm_controller/follow_joint_trajectory → execute

No manual offsets.  The TF tree (camera→EE extrinsic) does all the work.
If TF is wrong, fix the URDF/static_transform_publisher — not magic constants.

Usage:
    ros2 run <pkg> goto_pixel.py
    Click on the OpenCV window.  The arm moves the EE to that 3D point.
"""
from __future__ import annotations

import math
import os
import threading
import time

os.environ["QT_LOGGING_RULES"] = "qt.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─── Camera intrinsics (RealSense D435 defaults — replace with your calibration) ─
FX = 603.6312;  FY = 603.0632
CX = 319.0870;  CY = 236.3678


# ─── Frame names (must match your URDF / TF tree) ────────────────────────────────
BASE_FRAME = "j2s6s200_link_base"
CAM_FRAME  = "camera_color_optical_frame"
EE_LINK    = "j2s6s200_end_effector"
ARM_GROUP  = "arm"

ARM_JOINT_NAMES = [
    "j2s6s200_joint_1", "j2s6s200_joint_2", "j2s6s200_joint_3",
    "j2s6s200_joint_4", "j2s6s200_joint_5", "j2s6s200_joint_6",
]

# ─── Depth filtering ─────────────────────────────────────────────────────────────
DEPTH_PATCH_RADIUS = 4      # pixels: median filter over (2r+1)² patch
DEPTH_MIN_M        = 0.05   # reject closer than 5 cm
DEPTH_MAX_M        = 1.50   # reject farther than 1.5 m
DEPTH_MIN_VALID    = 4      # minimum valid pixels in patch

# ─── Motion parameters ───────────────────────────────────────────────────────────
IK_TIMEOUT_S       = 5      # MoveIt IK solver timeout
JOINT_VEL_LIMIT    = 0.15   # rad/s — max joint velocity for duration calc
MIN_MOVE_DURATION  = 3.0    # seconds — floor on trajectory duration
EXEC_TIMEOUT_S     = 30.0   # action server result timeout


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Geometry utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quat_to_rotation_matrix(q) -> np.ndarray:
    """ROS geometry_msgs Quaternion → 3×3 rotation matrix (Hamilton convention)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def backproject_pixel(u: int, v: int, depth: np.ndarray) -> np.ndarray | None:
    """
    Pinhole backprojection: pixel (u,v) + depth image → 3D point in camera frame.

    Uses a median filter over a small patch to reject depth noise / holes.
    Returns None if depth is invalid.
    """
    h, w = depth.shape[:2]
    r = DEPTH_PATCH_RADIUS

    v0, v1 = max(0, v - r), min(h, v + r + 1)
    u0, u1 = max(0, u - r), min(w, u + r + 1)

    patch = depth[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size < DEPTH_MIN_VALID:
        return None

    z = float(np.median(valid)) * 0.001          # mm → metres
    if not (DEPTH_MIN_M < z < DEPTH_MAX_M):
        return None

    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return np.array([x, y, z])


def cam_to_base(pt_cam: np.ndarray, tf_cam_to_base) -> np.ndarray:
    """Transform a 3D point from camera frame to base frame using a TF StampedTransform."""
    R = quat_to_rotation_matrix(tf_cam_to_base.transform.rotation)
    t = np.array([
        tf_cam_to_base.transform.translation.x,
        tf_cam_to_base.transform.translation.y,
        tf_cam_to_base.transform.translation.z,
    ])
    return R @ pt_cam + t


def shortest_angular_path(current: float, target: float) -> float:
    """Unwrap target so the joint takes the shortest path (handles ±2π wraps)."""
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROS2 Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GotoPixel(Node):
    """
    Click a pixel → arm moves EE there.

    Subscribes:
        /camera/camera/color/image_raw              (sensor_msgs/Image)
        /camera/camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image, 16UC1 mm)
        /joint_states                               (sensor_msgs/JointState)

    Clients:
        /compute_ik                                 (moveit_msgs/srv/GetPositionIK)
        /arm_controller/follow_joint_trajectory     (control_msgs/action/FollowJointTrajectory)
    """

    def __init__(self):
        super().__init__("goto_pixel")
        cb = ReentrantCallbackGroup()

        # ── state ─────────────────────────────────────────────────────────────
        self._lock      = threading.Lock()
        self._bridge    = CvBridge()
        self._color_img = None
        self._depth_img = None
        self._joints    = {n: 0.0 for n in ARM_JOINT_NAMES}
        self._busy      = False

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf = Buffer()
        TransformListener(self._tf_buf, self)

        # ── subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            Image, "/camera/camera/color/image_raw",
            self._on_color, 2, callback_group=cb)
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self._on_depth, 2, callback_group=cb)
        self.create_subscription(
            JointState, "/joint_states",
            self._on_joints, 10, callback_group=cb)

        # ── MoveIt IK service ─────────────────────────────────────────────────
        self._ik = self.create_client(GetPositionIK, "/compute_ik", callback_group=cb)
        self.get_logger().info("Waiting for /compute_ik ...")
        if not self._ik.wait_for_service(timeout_sec=30.0):
            raise RuntimeError("/compute_ik not available — is MoveIt running?")

        # ── trajectory action ─────────────────────────────────────────────────
        self._traj = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
            callback_group=cb)
        self.get_logger().info("Waiting for trajectory action server ...")
        if not self._traj.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("Trajectory action server not available")

        self.get_logger().info("GotoPixel ready — click on the OpenCV window.")

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _on_color(self, msg: Image):
        with self._lock:
            self._color_img = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    def _on_depth(self, msg: Image):
        with self._lock:
            self._depth_img = self._bridge.imgmsg_to_cv2(msg, "passthrough")

    def _on_joints(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._joints:
                    self._joints[name] = pos

    # ── Public entry point ────────────────────────────────────────────────────

    def goto_pixel(self, u: int, v: int):
        """
        Full pipeline: pixel → 3D → IK → execute.
        Runs on a background thread so ROS callbacks keep spinning.
        """
        if self._busy:
            self.get_logger().warn("Already moving — ignoring click")
            return
        threading.Thread(target=self._goto_pixel_impl, args=(u, v), daemon=True).start()

    def _goto_pixel_impl(self, u: int, v: int):
        self._busy = True
        try:
            self._pipeline(u, v)
        except Exception as e:
            self.get_logger().error(f"goto_pixel failed: {e}")
        finally:
            self._busy = False

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _pipeline(self, u: int, v: int):
        log = self.get_logger()

        # 1. Grab current depth + joints
        with self._lock:
            depth  = self._depth_img
            joints = dict(self._joints)
        if depth is None:
            log.error("No depth image yet"); return

        # 2. Backproject pixel → camera frame
        pt_cam = backproject_pixel(u, v, depth)
        if pt_cam is None:
            log.error(f"Invalid depth at pixel ({u}, {v})"); return
        log.info(f"pixel ({u},{v}) → cam [{pt_cam[0]:+.3f}, {pt_cam[1]:+.3f}, {pt_cam[2]:+.3f}] m")

        # 3. Transform camera → base frame
        try:
            tf = self._tf_buf.lookup_transform(
                BASE_FRAME, CAM_FRAME, Time(), Duration(seconds=2.0))
        except Exception as e:
            log.error(f"TF {CAM_FRAME}→{BASE_FRAME} failed: {e}"); return

        pt_base = cam_to_base(pt_cam, tf)
        log.info(f"base [{pt_base[0]:+.3f}, {pt_base[1]:+.3f}, {pt_base[2]:+.3f}] m")

        # 4. Get current EE orientation (preserve it — we only command position)
        try:
            tf_ee = self._tf_buf.lookup_transform(
                BASE_FRAME, EE_LINK, Time(), Duration(seconds=1.0))
            ee_quat = tf_ee.transform.rotation
        except Exception as e:
            log.error(f"TF {EE_LINK}→{BASE_FRAME} failed: {e}"); return

        # 5. Solve IK
        ik_joints = self._solve_ik(pt_base, ee_quat, joints)
        if ik_joints is None:
            return

        # 6. Execute trajectory
        cur_positions = [joints[n] for n in ARM_JOINT_NAMES]
        self._execute(cur_positions, ik_joints)

    # ── IK ────────────────────────────────────────────────────────────────────

    def _solve_ik(
        self,
        position_base: np.ndarray,
        orientation,                    # geometry_msgs/Quaternion
        current_joints: dict[str, float],
    ) -> list[float] | None:
        """Call MoveIt /compute_ik.  Returns unwrapped joint targets or None."""
        log = self.get_logger()

        req = GetPositionIK.Request()
        ik  = req.ik_request
        ik.group_name       = ARM_GROUP
        ik.ik_link_name     = EE_LINK
        ik.avoid_collisions = True
        ik.timeout.sec      = IK_TIMEOUT_S

        ik.pose_stamped.header.frame_id = BASE_FRAME
        ik.pose_stamped.header.stamp    = self.get_clock().now().to_msg()
        ik.pose_stamped.pose.position.x = float(position_base[0])
        ik.pose_stamped.pose.position.y = float(position_base[1])
        ik.pose_stamped.pose.position.z = float(position_base[2])
        ik.pose_stamped.pose.orientation = orientation

        ik.robot_state.joint_state.name     = ARM_JOINT_NAMES
        ik.robot_state.joint_state.position = [current_joints[n] for n in ARM_JOINT_NAMES]

        # synchronous-ish call via threading event (we're already off the executor thread)
        future = self._ik.call_async(req)
        done   = threading.Event()
        future.add_done_callback(lambda _: done.set())

        if not done.wait(timeout=IK_TIMEOUT_S + 5.0):
            log.error("IK service timed out"); return None

        result = future.result()
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            log.error(f"IK failed (code {result.error_code.val})"); return None

        # extract + unwrap
        sol_map = dict(zip(result.solution.joint_state.name,
                           result.solution.joint_state.position))
        target = []
        for name in ARM_JOINT_NAMES:
            cur = current_joints[name]
            tgt = sol_map.get(name, cur)
            target.append(shortest_angular_path(cur, tgt))

        log.info("IK solved")
        return target

    # ── Trajectory execution ──────────────────────────────────────────────────

    def _execute(self, current_pos: list[float], target_pos: list[float]) -> bool:
        """Build a two-point trajectory and send it to the arm controller."""
        log = self.get_logger()

        # duration from worst-case joint displacement at velocity limit
        max_delta = max(abs(c - t) for c, t in zip(current_pos, target_pos))
        duration  = max(MIN_MOVE_DURATION, max_delta / JOINT_VEL_LIMIT)
        log.info(f"Executing trajectory — {duration:.1f}s (max Δ = {max_delta:.3f} rad)")

        # build trajectory
        pt_start = JointTrajectoryPoint()
        pt_start.positions       = list(current_pos)
        pt_start.time_from_start = DurationMsg(sec=0, nanosec=0)

        sec_int = int(duration)
        pt_end  = JointTrajectoryPoint()
        pt_end.positions       = list(target_pos)
        pt_end.time_from_start = DurationMsg(
            sec=sec_int, nanosec=int((duration - sec_int) * 1e9))

        traj              = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt_start, pt_end]

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # send + wait
        done    = threading.Event()
        success = [False]

        def on_accepted(future):
            gh = future.result()
            if not gh.accepted:
                log.error("Trajectory rejected by controller")
                done.set()
                return
            log.info("Trajectory accepted — executing ...")
            gh.get_result_async().add_done_callback(on_result)

        def on_result(future):
            rc = future.result().result.error_code
            if rc == 0:
                log.info("Trajectory complete ✓")
                success[0] = True
            else:
                log.error(f"Trajectory failed (error_code={rc})")
            done.set()

        self._traj.send_goal_async(goal).add_done_callback(on_accepted)

        if not done.wait(timeout=EXEC_TIMEOUT_S):
            log.error("Trajectory execution timed out")
            return False
        return success[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OpenCV UI — minimal click handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ui(node: GotoPixel):
    win = "GotoPixel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(
        win,
        lambda ev, x, y, flags, _: node.goto_pixel(x, y)
            if ev == cv2.EVENT_LBUTTONDOWN else None)

    while rclpy.ok():
        with node._lock:
            frame = node._color_img
        if frame is not None:
            disp = frame.copy()
            h, w = disp.shape[:2]

            # crosshair
            cx, cy = w // 2, h // 2
            clr = (0, 255, 0) if not node._busy else (0, 0, 255)
            cv2.line(disp, (cx - 20, cy), (cx - 6, cy), clr, 1)
            cv2.line(disp, (cx + 6, cy), (cx + 20, cy),  clr, 1)
            cv2.line(disp, (cx, cy - 20), (cx, cy - 6),  clr, 1)
            cv2.line(disp, (cx, cy + 6),  (cx, cy + 20), clr, 1)

            label = "MOVING" if node._busy else "READY — click to move"
            cv2.putText(disp, label, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 1, cv2.LINE_AA)
            cv2.imshow(win, disp)

        if (cv2.waitKey(30) & 0xFF) in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    rclpy.init()
    node     = GotoPixel()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        run_ui(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()