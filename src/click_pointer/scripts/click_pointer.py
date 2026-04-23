#!/usr/bin/env python3
"""
click_pointer.py — Click a pixel, arm EE moves there.

Flow per click:
  1. pixel (u,v) + aligned depth → 3D point in camera frame
  2. TF → base frame  (point_base)
  3. TF → current EE rotation R
  4. tcp_target = point_base - R @ POINTER_OFFSET   (pointer-tip compensation)
  5. /plan_kinematic_path  with POSITION-ONLY constraint (no orientation lock)
     → OMPL finds any valid arm config that puts EE at tcp_target
  6. /arm_controller/follow_joint_trajectory → execute
"""

import threading
import time
from typing import List, Optional, Tuple

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
from geometry_msgs.msg import Pose, PointStamped
from moveit_msgs.msg import (Constraints, MotionPlanRequest,
                              PositionConstraint, BoundingVolume, RobotState)
from moveit_msgs.srv import GetMotionPlan
from sensor_msgs.msg import Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker

# ── Camera intrinsics ──────────────────────────────────────────────────────
# D435i @ 640×480 defaults. The node logs the true values on startup —
# update these if your resolution is different.
FX = 603.6312
FY = 603.0632
CX = 319.0870
CY = 236.3678

# ── Frames ─────────────────────────────────────────────────────────────────
BASE_FRAME   = 'j2s6s200_link_base'
EE_LINK      = 'j2s6s200_end_effector'
CAM_FRAME    = 'camera_color_optical_frame'
MOVEIT_GROUP = 'arm'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

# ── Tuning ─────────────────────────────────────────────────────────────────
# Vector from EE-frame origin to the actual pointer/fork tip (metres).
# Leave as zeros to home the EE origin to the clicked point first, then tune.
POINTER_OFFSET = np.array([0.0, 0.0, 0.0])

PLAN_TIME_S  = 5.0   # OMPL planning budget
VEL_SCALE    = 1.0   # use full joint velocity limits for planning
GOAL_TOL_M   = 0.03  # 3 cm position tolerance sphere
MAX_VEL_RAD  = 0.5   # rad/s cap for our 2-point trajectory
MOVE_DUR_S   = 3.0   # minimum move duration (seconds)


# ── Helpers ────────────────────────────────────────────────────────────────

def _wait_for_future(future, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while not future.done():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


# ── Node ───────────────────────────────────────────────────────────────────

class ClickPointerNode(Node):

    def __init__(self):
        super().__init__('click_pointer')
        self._cb = ReentrantCallbackGroup()

        self._bridge    = CvBridge()
        self._lock      = threading.Lock()
        self._color_img = None
        self._depth_img = None
        self._latest_js: Optional[JointState] = None
        self._busy      = False

        # TF
        self._tf_buf = Buffer()
        TransformListener(self._tf_buf, self)

        # RViz marker
        self._marker_pub = self.create_publisher(
            Marker, '/click_pointer/target_marker', 10)

        # Subscribers
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._color_cb, 10, callback_group=self._cb)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                                 self._depth_cb, 10, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states',
                                 self._js_cb, 10, callback_group=self._cb)

        # Motion planning service (/plan_kinematic_path from move_group)
        self._plan_client = self.create_client(
            GetMotionPlan, '/plan_kinematic_path', callback_group=self._cb)
        self.get_logger().info('Waiting for /plan_kinematic_path …')
        if not self._plan_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(
                'Timed out waiting for /plan_kinematic_path — is move_group running?')

        # Trajectory action
        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self._cb)
        self.get_logger().info('Waiting for trajectory controller …')
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for trajectory controller')

        self.get_logger().info(
            'click_pointer ready — click any pixel in the window to move there.')

    # ── Subscribers ─────────────────────────────────────────────────────

    def _color_cb(self, msg):
        with self._lock:
            self._color_img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_img = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')

    def _js_cb(self, msg):
        with self._lock:
            self._latest_js = msg

    # ── Mouse click ──────────────────────────────────────────────────────

    def on_click(self, u: int, v: int):
        if self._busy:
            self.get_logger().warn('Still moving — ignored.')
            return

        with self._lock:
            depth_img = self._depth_img
            js        = self._latest_js

        if depth_img is None:
            self.get_logger().warn('No depth image yet.')
            return
        if js is None:
            self.get_logger().warn('No joint states yet.')
            return
        if v >= depth_img.shape[0] or u >= depth_img.shape[1]:
            return

        raw = depth_img[v, u]
        if raw == 0:
            self.get_logger().warn(
                f'Depth = 0 at ({u},{v}) — surface not visible to depth sensor.')
            return

        z = float(raw) * 0.001  -   0.23  # mm → m
        x = (u - CX) * z / FX + 0.15  
        y = (v - CY) * z / FY +0.1

        # Camera → base frame
        p_cam = PointStamped()
        p_cam.header.frame_id = CAM_FRAME
        p_cam.header.stamp    = Time().to_msg()
        p_cam.point.x, p_cam.point.y, p_cam.point.z = x, y, z

        try:
            tf_cam = self._tf_buf.lookup_transform(
                BASE_FRAME, CAM_FRAME, Time(), timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF cam→base failed: {e}')
            return

        p_base = do_transform_point(p_cam, tf_cam)
        p_target = np.array([p_base.point.x, p_base.point.y, p_base.point.z])

        self.get_logger().info(
            f'Click ({u},{v})  depth={z:.3f}m  '
            f'→ base ({p_target[0]:+.3f},{p_target[1]:+.3f},{p_target[2]:+.3f})')

        # Current EE rotation from TF (for pointer offset compensation)
        try:
            tf_ee = self._tf_buf.lookup_transform(
                BASE_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF ee→base failed: {e}')
            return

        r = tf_ee.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()

        # EE must be at (p_target - R @ POINTER_OFFSET) so pointer tip lands on p_target
        tcp_target = p_target - R @ POINTER_OFFSET

        self._publish_marker(tcp_target)

        threading.Thread(
            target=self._execute_move,
            args=(tcp_target, js),
            daemon=True).start()

    # ── RViz marker ──────────────────────────────────────────────────────

    def _publish_marker(self, pos: np.ndarray):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = 'click_target', 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = float(pos[0]) 
        m.pose.position.y = float(pos[1]) 
        m.pose.position.z = float(pos[2]) 
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.05   # 5 cm sphere
        m.color   = ColorRGBA(r=1.0, g=0.3, b=0.0, a=0.9)
        m.lifetime.sec = 0   # persistent until cleared
        self._marker_pub.publish(m)

    def _publish_marker_clear(self):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = 'click_target', 0
        m.action          = Marker.DELETE
        self._marker_pub.publish(m)

    # ── Plan + execute ───────────────────────────────────────────────────

    def _execute_move(self, tcp_target: np.ndarray, current_js: JointState):
        self._busy = True
        try:
            print(f'\n>>> Target: ({tcp_target[0]:+.3f}, {tcp_target[1]:+.3f}, {tcp_target[2]:+.3f}) m'
                  f'  [check RViz for the orange sphere]')
            try:
                ans = input('>>> Move arm there? [y/N]: ').strip().lower()
            except EOFError:
                ans = 'n'
            if ans != 'y':
                print('>>> Skipped.')
                self._publish_marker_clear()
                return

            self.get_logger().info(
                f'Planning to ({tcp_target[0]:+.3f},{tcp_target[1]:+.3f},{tcp_target[2]:+.3f}) …')

            sphere = SolidPrimitive()
            sphere.type       = SolidPrimitive.SPHERE
            sphere.dimensions = [GOAL_TOL_M]

            centre = Pose()
            centre.position.x    = float(tcp_target[0])
            centre.position.y    = float(tcp_target[1])
            centre.position.z    = float(tcp_target[2])
            centre.orientation.w = 1.0

            bv = BoundingVolume()
            bv.primitives.append(sphere)
            bv.primitive_poses.append(centre)

            pos_con = PositionConstraint()
            pos_con.header.frame_id   = BASE_FRAME
            pos_con.link_name         = EE_LINK
            pos_con.constraint_region = bv
            pos_con.weight            = 1.0

            rs = RobotState()
            rs.joint_state = current_js

            mpr = MotionPlanRequest()
            mpr.group_name                      = MOVEIT_GROUP
            mpr.start_state                     = rs
            mpr.goal_constraints                = [Constraints(position_constraints=[pos_con])]
            mpr.num_planning_attempts           = 10
            mpr.allowed_planning_time           = PLAN_TIME_S
            mpr.max_velocity_scaling_factor     = VEL_SCALE
            mpr.max_acceleration_scaling_factor = VEL_SCALE

            req = GetMotionPlan.Request()
            req.motion_plan_request = mpr

            future = self._plan_client.call_async(req)
            if not _wait_for_future(future, PLAN_TIME_S + 5.0):
                self.get_logger().error('Planning timed out.')
                return

            res = future.result()
            if res is None or res.motion_plan_response.error_code.val != 1:
                code = res.motion_plan_response.error_code.val if res else 'None'
                self.get_logger().error(
                    f'Planning failed (code={code}) — outside workspace (~0.85 m from arm base).')
                return

            planned = res.motion_plan_response.trajectory.joint_trajectory
            if not planned.points:
                self.get_logger().error('Planner returned empty trajectory.')
                return

            # Extract only the GOAL joint config from the planned trajectory.
            # We send it as a simple 2-point trajectory (current → goal) at our
            # own speed — this avoids PATH_TOLERANCE_VIOLATED that happens when
            # the velocity bridge tries to track 100+ tightly-timed waypoints.
            goal_pt   = planned.points[-1]
            goal_names = planned.joint_names
            goal_map  = dict(zip(goal_names, goal_pt.positions))

            self.get_logger().info(
                f'Plan OK ({len(planned.points)} wpts) → sending 2-pt trajectory …')

            self._send_two_point(goal_map, current_js)

        except Exception as exc:
            self.get_logger().error(f'Exception: {exc}')
        finally:
            self._busy = False

    def _send_two_point(self, goal_map: dict, current_js: JointState):
        names     = [n for n in ARM_JOINT_NAMES if n in goal_map]
        targets   = [goal_map[n] for n in names]
        cur_map   = dict(zip(current_js.name, current_js.position))
        cur_pos   = [float(cur_map.get(n, t)) for n, t in zip(names, targets)]

        duration = MOVE_DUR_S
        for n, t in zip(names, targets):
            if n in cur_map:
                duration = max(duration, abs(t - float(cur_map[n])) / MAX_VEL_RAD)

        sec  = int(duration)
        nsec = int((duration - sec) * 1e9)

        from trajectory_msgs.msg import JointTrajectoryPoint
        pt0 = JointTrajectoryPoint()
        pt0.positions       = cur_pos
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

        pt1 = JointTrajectoryPoint()
        pt1.positions       = targets
        pt1.time_from_start = DurationMsg(sec=sec, nanosec=nsec)

        traj              = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = names
        traj.points       = [pt0, pt1]

        goal            = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        f = self._traj_client.send_goal_async(goal)
        if not _wait_for_future(f, 5.0):
            self.get_logger().error('send_goal timed out.')
            return
        gh = f.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('Trajectory rejected.')
            return

        res_f = gh.get_result_async()
        if not _wait_for_future(res_f, duration + 15.0):
            self.get_logger().error('Execution timed out.')
            return

        rw = res_f.result()
        if rw and rw.result.error_code == 0:
            self.get_logger().info('Move complete.')
            self._publish_marker_clear()
        else:
            self.get_logger().warn(
                f'Finished with error_code={rw.result.error_code if rw else "None"}')

    # ── Frame for display ────────────────────────────────────────────────

    def get_frame(self):
        with self._lock:
            return self._color_img


# ── OpenCV UI ──────────────────────────────────────────────────────────────

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
            label = 'PLANNING / MOVING...' if node._busy else 'Click to point'
            color = (0, 0, 255) if node._busy else (0, 255, 0)
            cv2.putText(disp, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow(WIN, disp)
        key = cv2.waitKey(33)
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()


# ── Entry point ────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ClickPointerNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        run_ui(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
