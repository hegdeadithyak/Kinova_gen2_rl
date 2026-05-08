#!/usr/bin/env python3
"""
click_pointer.py — Phased Algorithmic Joint Stepping with Decision Logging

Flow per click:
  1. Pixel (u,v) + depth → 3D point in camera frame
  2. Apply EE offset: target_y -= CAM_Y_OFFSET  (camera is 9 cm above EE)
  3. Get EE start position in camera frame via TF
  4. Execute phased algorithm in camera frame:
     - Phase 1: Adjust Y until error < 0.05m (Steps J2)
     - Phase 2: Adjust X until error < 0.05m (Steps J3)
     - Phase 3: Adjust Z until error < 0.05m (Steps J1 & J5)
  5. Position tracked manually: fraction = clamp(|error| / CART_DELTA, 1.0),
     joint step = fraction * STEP_RADS, expected Cartesian move = fraction * CART_DELTA
"""

import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── Camera intrinsics ──────────────────────────────────────────────────────
FX = 603.6312
FY = 603.0632
CX = 319.0870
CY = 236.3678

# ── Frames & Groups ────────────────────────────────────────────────────────
EE_LINK   = 'j2s6s200_end_effector'
CAM_FRAME = 'camera_color_optical_frame'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

# ── Camera-to-EE offset ────────────────────────────────────────────────────
CAM_Y_OFFSET = 0.09   # Camera is 9 cm above EE in camera-frame Y

# ── Stepping parameters ────────────────────────────────────────────────────
STEP_RADS      = 0.05   # Maximum joint radians per tick
CART_DELTA_X   = 0.05   # Cartesian metres per STEP_RADS for X axis (calibrated)
CART_DELTA_Y   = 0.2    # Cartesian metres per STEP_RADS for Y axis
CART_DELTA_Z   = 0.01   # Cartesian metres per STEP_RADS for Z axis (calibrated)
TICK_DUR_S     = 0.6    # Seconds to let the physical joint settle
ERR_TOL_M      = 0.05   # 5 cm tolerance for phase completion


class ClickPointerNode(Node):

    def __init__(self):
        super().__init__('click_pointer')
        self._cb = ReentrantCallbackGroup()

        self._bridge    = CvBridge()
        self._lock      = threading.Lock()
        self._color_img = None
        self._depth_img = None
        self._busy      = False

        self._current_positions = {name: 0.0 for name in ARM_JOINT_NAMES}

        self._tf_buf = Buffer()
        TransformListener(self._tf_buf, self)

        # Subscribers
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._color_cb, 10, callback_group=self._cb)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                                 self._depth_cb, 10, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states',
                                 self._js_cb, 10, callback_group=self._cb)

        # Joint Trajectory Action
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self._cb,
        )
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
        if raw == 0:
            return

        self.get_logger().info("\n==================================================")
        self.get_logger().info(f"[DEBUG] NEW CLICK REGISTERED")
        self.get_logger().info(f"[DEBUG] Raw Pixel: (u={u}, v={v}), Depth z_raw={float(raw)*0.001:.3f}m")

        # Pixel + depth → 3D in camera frame (standard pinhole)
        z = float(raw) * 0.001
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        self.get_logger().info(f"[DEBUG] 1. Click Point in Camera Frame: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

        # Target for EE: camera is CAM_Y_OFFSET above EE → shift target down in cam Y
        endpoint = np.array([x, y - CAM_Y_OFFSET, z])
        self.get_logger().info(
            f"[DEBUG] 2. EE Target in Camera Frame (y-={CAM_Y_OFFSET}): "
            f"X={endpoint[0]:.3f}, Y={endpoint[1]:.3f}, Z={endpoint[2]:.3f}"
        )

        # EE start position in camera frame
        try:
            tf_ee_cam = self._tf_buf.lookup_transform(
                CAM_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return

        startpoint = np.array([
            tf_ee_cam.transform.translation.x,
            tf_ee_cam.transform.translation.y,
            tf_ee_cam.transform.translation.z,
        ])
        self.get_logger().info(
            f"[DEBUG] 3. EE Start in Camera Frame: "
            f"X={startpoint[0]:.3f}, Y={startpoint[1]:.3f}, Z={startpoint[2]:.3f}"
        )

        dx = endpoint[0] - startpoint[0]
        dy = endpoint[1] - startpoint[1]
        dz = endpoint[2] - startpoint[2]
        self.get_logger().info(f"[DEBUG] 4. Error: dx={dx:+.3f}  dy={dy:+.3f}  dz={dz:+.3f}")

        ans = input(
            f'\n>>> Target (cam frame): ({endpoint[0]:+.3f}, {endpoint[1]:+.3f}, {endpoint[2]:+.3f}). '
            f'Begin sequence? [y/N]: '
        ).strip().lower()
        if ans != 'y':
            print('>>> Skipped.')
            return

        threading.Thread(target=self.perform, args=(startpoint, endpoint), daemon=True).start()

    # ── The Phased Algorithm ──────────────────────────────────────────────
    def perform(self, startpoint: np.ndarray, endpoint: np.ndarray):
        """
        Phased approach entirely in camera frame.

        Position is tracked manually without querying the arm:
          fraction = clamp(|error| / CART_DELTA_PER_STEP, 0, 1)
          joint step sent  = fraction * STEP_RADS
          expected EE move = fraction * CART_DELTA_PER_STEP metres

        Tune CART_DELTA_PER_STEP until the manual tracking matches reality.
        """
        self._busy = True

        try:
            currpoint = startpoint.copy()
            dx = endpoint[0] - currpoint[0]
            dy = endpoint[1] - currpoint[1]
            dz = endpoint[2] - currpoint[2]

            self.get_logger().info("\n==================================================")
            self.get_logger().info(f"[START] startpoint=({startpoint[0]:+.3f},{startpoint[1]:+.3f},{startpoint[2]:+.3f})")
            self.get_logger().info(f"[START] endpoint=  ({endpoint[0]:+.3f},{endpoint[1]:+.3f},{endpoint[2]:+.3f})")
            self.get_logger().info(f"[START] needed: dx={dx:+.3f}  dy={dy:+.3f}  dz={dz:+.3f}")

            # ── Phase 1: Match Y (J2) ─────────────────────────────────────
            self.get_logger().info("\n=== Phase 1: Adjusting Y Axis (J2) ===")
            while abs(dy) > ERR_TOL_M:
                fraction = min(abs(dy) / CART_DELTA_Y, 1.0)
                step = fraction * STEP_RADS
                self.get_logger().info(
                    f"  [EVAL] dy={dy:+.3f}  curr_Y={currpoint[1]:+.3f}  "
                    f"target_Y={endpoint[1]:+.3f}  fraction={fraction:.2f}"
                )
                if dy > 0:
                    self.get_logger().info(f"  [CHOICE] dy positive → decrease_j2")
                    self.decrease_j2(step)
                    currpoint[1] += fraction * CART_DELTA_Y
                else:
                    self.get_logger().info(f"  [CHOICE] dy negative → increase_j2")
                    self.increase_j2(step)
                    currpoint[1] -= fraction * CART_DELTA_Y
                time.sleep(TICK_DUR_S)
                dy = endpoint[1] - currpoint[1]
                self.get_logger().info(f"  [RESULT] expected_Y={currpoint[1]:+.3f}  new dy={dy:+.3f}")
            self.get_logger().info(f"=== Phase 1 Complete (dy={dy:+.3f}) ===")

            # ── Phase 2: Match X (J3) ─────────────────────────────────────
            self.get_logger().info("\n=== Phase 2: Adjusting X Axis (J3) ===")
            while abs(dx) > ERR_TOL_M:
                fraction = min(abs(dx) / CART_DELTA_X, 1.0)
                step = fraction * STEP_RADS
                self.get_logger().info(
                    f"  [EVAL] dx={dx:+.3f}  curr_X={currpoint[0]:+.3f}  "
                    f"target_X={endpoint[0]:+.3f}  fraction={fraction:.2f}"
                )
                if dx < 0:
                    self.get_logger().info(f"  [CHOICE] dx negative → increase_j3")
                    self.decrease_j3(step)
                    currpoint[0] -= fraction * CART_DELTA_X
                else:
                    self.get_logger().info(f"  [CHOICE] dx positive → decrease_j3")
                    self.increase_j3(step)
                    currpoint[0] += fraction * CART_DELTA_X
                time.sleep(TICK_DUR_S)
                dx = endpoint[0] - currpoint[0]
                self.get_logger().info(f"  [RESULT] expected_X={currpoint[0]:+.3f}  new dx={dx:+.3f}")
            self.get_logger().info(f"=== Phase 2 Complete (dx={dx:+.3f}) ===")

            # ── Phase 3: Approach forward (J1 + J5) ───────────────────────
            self.get_logger().info("\n=== Phase 3: Adjusting Z Axis (J1 + J5) ===")
            while dz > ERR_TOL_M:
                fraction = min(dz / CART_DELTA_Z, 1.0)
                step = fraction * STEP_RADS
                self.get_logger().info(
                    f"  [EVAL] dz={dz:+.3f}  curr_Z={currpoint[2]:+.3f}  "
                    f"target_Z={endpoint[2]:+.3f}  fraction={fraction:.2f}"
                )
                self.get_logger().info(f"  [CHOICE] increase_j1 + increase_j5/2 (push forward)")
                self.increase_j1(step)
                self.increase_j4(step / 2.0)
                currpoint[2] += fraction * CART_DELTA_Z
                time.sleep(TICK_DUR_S)
                dz = endpoint[2] - currpoint[2]
                self.get_logger().info(f"  [RESULT] expected_Z={currpoint[2]:+.3f}  new dz={dz:+.3f}")
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
        """
        with self._lock:
            current_pos = [self._current_positions[name] for name in ARM_JOINT_NAMES]

        target_pos = list(current_pos)
        target_pos[joint_idx] += delta_rad

        pt0 = JointTrajectoryPoint()
        pt0.positions = current_pos
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

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

    def increase_j1(self, step): self._dispatch_step(0,  abs(step))
    def decrease_j1(self, step): self._dispatch_step(0, -abs(step))
    def increase_j2(self, step): self._dispatch_step(1,  abs(step))
    def decrease_j2(self, step): self._dispatch_step(1, -abs(step))
    def increase_j3(self, step): self._dispatch_step(2,  abs(step))
    def decrease_j3(self, step): self._dispatch_step(2, -abs(step))
    def increase_j4(self, step): self._dispatch_step(3,  abs(step))
    def decrease_j4(self, step): self._dispatch_step(3, -abs(step))
    def increase_j5(self, step): self._dispatch_step(4,  abs(step))
    def decrease_j5(self, step): self._dispatch_step(4, -abs(step))
    def increase_j6(self, step): self._dispatch_step(5,  abs(step))
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
