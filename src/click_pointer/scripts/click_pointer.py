#!/usr/bin/env python3
"""
click_pointer.py — Phased Algorithmic Joint Stepping with Decision Logging

Flow per click:
  1. Pixel (u,v) + depth → 3D point in camera frame
  2. TF → base frame (endpoint)
  3. execute phased algorithm:
     - Phase 1: Adjust Y until error < 0.05m (Steps J2)
     - Phase 2: Adjust X until error < 0.05m (Steps J3)
     - Phase 3: Adjust Z until error < 0.05m (Steps J1 & J5)
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── Camera intrinsics ──────────────────────────────────────────────────────
FX = 603.6312
FY = 603.0632
CX = 319.0870
CY = 236.3678

# ── Frames & Groups ────────────────────────────────────────────────────────
BASE_FRAME   = 'j2s6s200_link_base'
EE_LINK      = 'j2s6s200_end_effector'
CAM_FRAME    = 'camera_color_optical_frame'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

POINTER_OFFSET = np.array([0.0, 0.0, 0.0])

# Tuning parameters for the physical joint stepping
STEP_RADS = 0.05    # Radians to move per algorithmic tick
TICK_DUR_S = 0.6    # Seconds to allow the physical joint to reach its step before looping
ERR_TOL_M = 0.05    # 5cm Cartesian tolerance for phase completion


class ClickPointerNode(Node):

    def __init__(self):
        super().__init__('click_pointer')
        self._cb = ReentrantCallbackGroup()

        self._bridge    = CvBridge()
        self._lock      = threading.Lock()
        self._color_img = None
        self._depth_img = None
        self._busy      = False

        # Live tracking of joint positions and SDK cartesian pose
        self._current_positions = {name: 0.0 for name in ARM_JOINT_NAMES}
        self._latest_cart_pos: Optional[np.ndarray] = None  # from /kinova/cartesian_pose

        self._tf_buf = Buffer()
        TransformListener(self._tf_buf, self)

        # Subscribers
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._color_cb, 10, callback_group=self._cb)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10, callback_group=self._cb)
        self.create_subscription(PoseStamped, '/kinova/cartesian_pose', self._cart_cb, 10, callback_group=self._cb)

        # Joint Trajectory Action (Used purely for short position dispatches)
        self._traj_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory', callback_group=self._cb)
        self.get_logger().info('Waiting for trajectory controller ...')
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for trajectory controller')

        self.get_logger().info('Algorithmic Click Pointer Ready. Click to begin sequence.')

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _color_cb(self, msg):
        with self._lock: self._color_img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        with self._lock: self._depth_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _js_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._current_positions:
                    self._current_positions[name] = pos

    def _cart_cb(self, msg: PoseStamped):
        with self._lock:
            self._latest_cart_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])

    # ── Click Handling & Math ─────────────────────────────────────────────
    def on_click(self, u: int, v: int):
        if self._busy:
            self.get_logger().warn('Arm is currently running a feeding sequence. Ignored.')
            return

        with self._lock:
            depth_img = self._depth_img

        if depth_img is None:
            return

        raw = depth_img[v, u]
        if raw == 0: return

        self.get_logger().info("\n==================================================")
        self.get_logger().info(f"[DEBUG] NEW CLICK REGISTERED")
        self.get_logger().info(f"[DEBUG] Raw Pixel: (u={u}, v={v}), Depth z_raw={float(raw)*0.001:.3f}m")

        # Depth to 3D Math
        z = float(raw) * 0.001 - 0.23
        x = (u - CX) * z / FX + 0.15  
        y = (v - CY) * z / FY + 0.1

        from geometry_msgs.msg import PointStamped
        p_cam = PointStamped()
        p_cam.header.frame_id = CAM_FRAME
        p_cam.point.x, p_cam.point.y, p_cam.point.z = x, y, z

        self.get_logger().info(f"[DEBUG] 1. Calculated Point in Camera Frame: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

        try:
            tf_cam = self._tf_buf.lookup_transform(BASE_FRAME, CAM_FRAME, Time(), timeout=Duration(seconds=1.0))
            tf_ee = self._tf_buf.lookup_transform(BASE_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return

        # Grab exact starting point
        startpoint = np.array([
            tf_ee.transform.translation.x, 
            tf_ee.transform.translation.y, 
            tf_ee.transform.translation.z
        ])
        self.get_logger().info(f"[DEBUG] 2. Current End Effector (Start Point): X={startpoint[0]:.3f}, Y={startpoint[1]:.3f}, Z={startpoint[2]:.3f}")

        # Calculate pure Cartesian endpoint
        p_base = do_transform_point(p_cam, tf_cam)
        p_target = np.array([p_base.point.x, p_base.point.y, p_base.point.z])
        self.get_logger().info(f"[DEBUG] 3. Target in Base Frame: X={p_target[0]:.3f}, Y={p_target[1]:.3f}, Z={p_target[2]:.3f}")

        r = tf_ee.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        endpoint = p_target - R @ POINTER_OFFSET
        
        self.get_logger().info(f"[DEBUG] 4. Final Endpoint (after offset): X={endpoint[0]:.3f}, Y={endpoint[1]:.3f}, Z={endpoint[2]:.3f}")

        ans = input(f'\n>>> Detected target: ({endpoint[0]:+.3f}, {endpoint[1]:+.3f}, {endpoint[2]:+.3f}). Begin sequence? [y/N]: ').strip().lower()
        if ans != 'y':
            print('>>> Skipped.')
            return

        # Dispatch the algorithm to a background thread
        threading.Thread(target=self.perform, args=(startpoint, endpoint), daemon=True).start()

    # ── The Phased Algorithm ──────────────────────────────────────────────
    def get_currpoint(self, sdk_start: np.ndarray, startpoint: np.ndarray) -> np.ndarray:
        """
        Return current EE position in BASE_FRAME.

        Priority:
          1. TF lookup — the canonical source when robot_state_publisher is
             running from the official kinova_arm_driver stack.  Updated at
             the driver's joint-state publish rate (~100 Hz) and reflects real
             hardware position via FK.
          2. SDK cartesian delta — fallback for setups where kinova_sdk_node
             publishes /kinova/cartesian_pose instead of the official driver.
             Uses delta from start to avoid absolute frame offset issues.
          3. Hold last known point — both sources failed.

        NOTE: do NOT run kinova_sdk_node alongside kinova_arm_driver.
        They share the same USB device; the SDK node will receive stale data
        and /kinova/cartesian_pose will appear frozen.
        """
        # ── 1. TF (primary) ──────────────────────────────────────────
        try:
            tf_ee = self._tf_buf.lookup_transform(BASE_FRAME, EE_LINK, Time())
            return np.array([
                tf_ee.transform.translation.x,
                tf_ee.transform.translation.y,
                tf_ee.transform.translation.z,
            ])
        except Exception:
            pass

        # ── 2. SDK cartesian delta (fallback) ─────────────────────────
        with self._lock:
            cart = self._latest_cart_pos
        if cart is not None and sdk_start is not None:
            return startpoint + (cart - sdk_start)

        # ── 3. Hold ───────────────────────────────────────────────────
        self.get_logger().warn('TF and SDK pose both unavailable — holding last known point.')
        return startpoint

    def perform(self, startpoint: np.ndarray, endpoint: np.ndarray):
        self._busy = True

        try:
            # Snapshot SDK cartesian pose RIGHT NOW — used as delta anchor.
            # All get_currpoint() calls will return startpoint + (sdk_now - sdk_here).
            # This means absolute coordinate frame mismatches (origin offsets between
            # the SDK frame and TF base frame) are irrelevant — only deltas matter.
            with self._lock:
                sdk_start = self._latest_cart_pos.copy() if self._latest_cart_pos is not None else None

            if sdk_start is None:
                self.get_logger().warn(
                    '/kinova/cartesian_pose not received — falling back to TF for position tracking.'
                )

            currpoint = startpoint.copy()
            dx = endpoint[0] - currpoint[0]
            dy = endpoint[1] - currpoint[1]
            dz = endpoint[2] - currpoint[2]

            self.get_logger().info("\n==================================================")
            self.get_logger().info(f"[START] startpoint=({startpoint[0]:+.3f},{startpoint[1]:+.3f},{startpoint[2]:+.3f})")
            self.get_logger().info(f"[START] endpoint=  ({endpoint[0]:+.3f},{endpoint[1]:+.3f},{endpoint[2]:+.3f})")
            if sdk_start is not None:
                self.get_logger().info(f"[START] sdk_start= ({sdk_start[0]:+.3f},{sdk_start[1]:+.3f},{sdk_start[2]:+.3f})")
            else:
                self.get_logger().info("[START] sdk_start= None (using TF fallback)")
            self.get_logger().info(f"[START] needed: dx={dx:+.3f}  dy={dy:+.3f}  dz={dz:+.3f}")

            # ── Phase 1: Lift / lower to match Y ──────────────────────────
            self.get_logger().info("\n=== Phase 1: Adjusting Y Axis (J2) ===")
            while abs(dy) > ERR_TOL_M:
                self.get_logger().info(f"  [EVAL] dy={dy:+.3f}  curr_Y={currpoint[1]:+.3f}  target_Y={endpoint[1]:+.3f}")
                if dy > 0:
                    self.get_logger().info(f"  [CHOICE] dy positive → decrease_j2 (lift EE)")
                    self.decrease_j2(STEP_RADS)
                else:
                    self.get_logger().info(f"  [CHOICE] dy negative → increase_j2 (lower EE)")
                    self.increase_j2(STEP_RADS)
                time.sleep(TICK_DUR_S)
                currpoint = self.get_currpoint(sdk_start, startpoint)
                dy = endpoint[1] - currpoint[1]
                self.get_logger().info(f"  [RESULT] curr_Y={currpoint[1]:+.3f}  new dy={dy:+.3f}")
            self.get_logger().info(f"=== Phase 1 Complete (dy={dy:+.3f}) ===")

            # ── Phase 2: Shift left/right to match X ─────────────────────
            self.get_logger().info("\n=== Phase 2: Adjusting X Axis (J3) ===")
            while abs(dx) > ERR_TOL_M:
                self.get_logger().info(f"  [EVAL] dx={dx:+.3f}  curr_X={currpoint[0]:+.3f}  target_X={endpoint[0]:+.3f}")
                if dx < 0:
                    self.get_logger().info(f"  [CHOICE] dx negative → increase_j3 (left)")
                    self.increase_j3(STEP_RADS)
                else:
                    self.get_logger().info(f"  [CHOICE] dx positive → decrease_j3 (right)")
                    self.decrease_j3(STEP_RADS)
                time.sleep(TICK_DUR_S)
                currpoint = self.get_currpoint(sdk_start, startpoint)
                dx = endpoint[0] - currpoint[0]
                self.get_logger().info(f"  [RESULT] curr_X={currpoint[0]:+.3f}  new dx={dx:+.3f}")
            self.get_logger().info(f"=== Phase 2 Complete (dx={dx:+.3f}) ===")

            # ── Phase 3: Approach forward to match Z ─────────────────────
            # dz > 0 means target is further away (feeding task only goes forward)
            self.get_logger().info("\n=== Phase 3: Adjusting Z Axis (J1 + J5) ===")
            while dz > ERR_TOL_M:
                self.get_logger().info(f"  [EVAL] dz={dz:+.3f}  curr_Z={currpoint[2]:+.3f}  target_Z={endpoint[2]:+.3f}")
                self.get_logger().info(f"  [CHOICE] increase_j1 + increase_j5/2 (push forward)")
                self.increase_j1(STEP_RADS)
                self.increase_j5(STEP_RADS / 2.0)
                time.sleep(TICK_DUR_S)
                currpoint = self.get_currpoint(sdk_start, startpoint)
                dz = endpoint[2] - currpoint[2]
                self.get_logger().info(f"  [RESULT] curr_Z={currpoint[2]:+.3f}  new dz={dz:+.3f}")
            self.get_logger().info(f"=== Phase 3 Complete (dz={dz:+.3f}) ===")

            print("\n>>> fed the patient")
            self.get_logger().info("==================================================\n")

        except Exception as e:
            self.get_logger().error(f"Error during sequence: {e}")
        finally:
            self._busy = False

    # ── Direct Joint Control API ──────────────────────────────────────────
    def _dispatch_step(self, joint_idx: int, delta_rad: float):
        """
        Send a 2-point trajectory that steps one joint by delta_rad.

        Two points are required because the velocity-based trajectory executor
        derives joint velocity from (pt1.positions - pt0.positions) / dt.
        A single-point trajectory would produce zero velocity for all joints.

        The velocity produced = delta_rad / TICK_DUR_S for the target joint,
        and 0 for every other joint — regardless of what _current_positions
        holds, because only the delta between pt0 and pt1 matters.
        """
        with self._lock:
            current_pos = [self._current_positions[name] for name in ARM_JOINT_NAMES]

        target_pos = list(current_pos)
        target_pos[joint_idx] += delta_rad

        # pt0: current position at t=0  (velocity executor starts here)
        pt0 = JointTrajectoryPoint()
        pt0.positions = current_pos
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

        # pt1: target position at t=TICK_DUR_S  (executor interpolates velocity)
        pt1 = JointTrajectoryPoint()
        pt1.positions = target_pos
        pt1.time_from_start = DurationMsg(sec=0, nanosec=int(TICK_DUR_S * 1e9))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ARM_JOINT_NAMES
        traj.points = [pt0, pt1]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self._traj_client.send_goal_async(goal)

    def increase_j1(self, step): self._dispatch_step(0, abs(step))
    def decrease_j1(self, step): self._dispatch_step(0, -abs(step))
    def increase_j2(self, step): self._dispatch_step(1, abs(step))
    def decrease_j2(self, step): self._dispatch_step(1, -abs(step))
    def increase_j3(self, step): self._dispatch_step(2, abs(step))
    def decrease_j3(self, step): self._dispatch_step(2, -abs(step))
    def increase_j4(self, step): self._dispatch_step(3, abs(step))
    def decrease_j4(self, step): self._dispatch_step(3, -abs(step))
    def increase_j5(self, step): self._dispatch_step(4, abs(step))
    def decrease_j5(self, step): self._dispatch_step(4, -abs(step))
    def increase_j6(self, step): self._dispatch_step(5, abs(step))
    def decrease_j6(self, step): self._dispatch_step(5, -abs(step))

    # ── UI ────────────────────────────────────────────────────────────────
    def get_frame(self):
        with self._lock: return self._color_img

def run_ui(node: ClickPointerNode):
    WIN = 'Color (click to point)'
    cv2.namedWindow(WIN)

    def mouse_cb(event, u, v, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            node.on_click(u, v)

    cv2.setMouseCallback(WIN, mouse_cb)

    while rclpy.ok():
        frame = node.get_frame()
        if frame is not None:
            disp  = frame.copy()
            label = 'RUNNING SEQUENCE...' if node._busy else 'Click to point'
            color = (0, 0, 255) if node._busy else (0, 255, 0)
            cv2.putText(disp, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow(WIN, disp)
        if cv2.waitKey(33) in (ord('q'), 27): break

    cv2.destroyAllWindows()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ClickPointerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try: run_ui(node)
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()