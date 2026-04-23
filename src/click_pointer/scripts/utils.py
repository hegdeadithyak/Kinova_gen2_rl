#!/usr/bin/env python3
"""
click_pointer.py — Pure Position-Based Kinematics with Enhanced Debug Logging

Flow per click:
  1. Pixel (u,v) + depth → 3D point in camera frame
  2. TF → base frame (tcp_target)
  3. /compute_ik → Solves for the exact 6 joint angles (in radians) needed to reach the target.
  4. /arm_controller/follow_joint_trajectory → Sends a strict positional command to the joints.
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
from moveit_msgs.msg import RobotState, PositionIKRequest
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import ColorRGBA
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
#!/usr/bin/env python3
"""
kinova_joint_api.py — Programmatic Joint Position API

Provides explicit, callable functions to step individual Kinova joints.
Designed to be imported and used inside custom algorithms.
"""

import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

class KinovaJointAPI(Node):
    def __init__(self, move_duration_s=0.5):
        super().__init__('kinova_joint_api')
        self._cb_group = ReentrantCallbackGroup()
        self._move_duration_s = move_duration_s
        
        self._current_positions = {name: 0.0 for name in ARM_JOINT_NAMES}
        self._state_lock = threading.Lock()

        # Track live positions
        self.create_subscription(
            JointState, 
            '/joint_states', 
            self._joint_state_cb, 
            10, 
            callback_group=self._cb_group
        )

        # Action Client for execution
        self._traj_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/arm_controller/follow_joint_trajectory', 
            callback_group=self._cb_group
        )
        
        self.get_logger().info('Waiting for trajectory controller...')
        if not self._traj_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('Trajectory controller not found!')
            
        self.get_logger().info('Kinova Joint API Ready.')

    def _joint_state_cb(self, msg: JointState):
        """Keep an exact record of the arm's current physical state in radians."""
        with self._state_lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._current_positions:
                    self._current_positions[name] = pos

    def _dispatch_step(self, joint_idx: int, delta_rad: float):
        """Internal helper to calculate and send the position command."""
        joint_name = ARM_JOINT_NAMES[joint_idx]
        
        with self._state_lock:
            current_pos = [self._current_positions[name] for name in ARM_JOINT_NAMES]
        
        target_pos = list(current_pos)
        target_pos[joint_idx] += delta_rad

        self.get_logger().info(f"Stepping {joint_name} to {target_pos[joint_idx]:.3f} rad")

        pt = JointTrajectoryPoint()
        pt.positions = target_pos
        pt.time_from_start = DurationMsg(sec=0, nanosec=int(self._move_duration_s * 1e9))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ARM_JOINT_NAMES
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Fire and forget
        self._traj_client.send_goal_async(goal)

    # ── Callable Joint Functions ─────────────────────────────────────────────

    # JOINT 1
    def increase_j1(self, step_size: float):
        self._dispatch_step(0, abs(step_size))
        
    def decrease_j1(self, step_size: float):
        self._dispatch_step(0, -abs(step_size))

    # JOINT 2
    def increase_j2(self, step_size: float):
        self._dispatch_step(1, abs(step_size))
        
    def decrease_j2(self, step_size: float):
        self._dispatch_step(1, -abs(step_size))

    # JOINT 3
    def increase_j3(self, step_size: float):
        self._dispatch_step(2, abs(step_size))
        
    def decrease_j3(self, step_size: float):
        self._dispatch_step(2, -abs(step_size))

    # JOINT 4
    def increase_j4(self, step_size: float):
        self._dispatch_step(3, abs(step_size))
        
    def decrease_j4(self, step_size: float):
        self._dispatch_step(3, -abs(step_size))

    # JOINT 5
    def increase_j5(self, step_size: float):
        self._dispatch_step(4, abs(step_size))
        
    def decrease_j5(self, step_size: float):
        self._dispatch_step(4, -abs(step_size))

    # JOINT 6
    def increase_j6(self, step_size: float):
        self._dispatch_step(5, abs(step_size))
        
    def decrease_j6(self, step_size: float):
        self._dispatch_step(5, -abs(step_size))


# ── Camera intrinsics ──────────────────────────────────────────────────────
FX = 603.6312
FY = 603.0632
CX = 319.0870
CY = 236.3678

# ── Frames & Groups ────────────────────────────────────────────────────────
BASE_FRAME   = 'j2s6s200_link_base'
EE_LINK      = 'j2s6s200_end_effector'
CAM_FRAME    = 'camera_color_optical_frame'
MOVEIT_GROUP = 'arm'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

POINTER_OFFSET = np.array([0.0, 0.0, 0.0])
MOVE_DUR_S   = 4.0   # Fixed duration for the arm to reach the position goal


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

        self._tf_buf = Buffer()
        TransformListener(self._tf_buf, self)

        self._marker_pub = self.create_publisher(Marker, '/click_pointer/target_marker', 10)

        # Subscribers
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._color_cb, 10, callback_group=self._cb)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10, callback_group=self._cb)

        self._ik_client = self.create_client(GetPositionIK, '/compute_ik', callback_group=self._cb)
        self.get_logger().info('Waiting for /compute_ik ...')
        if not self._ik_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for /compute_ik service.')

        # Joint Trajectory Action (Used purely for positional dispatch)
        self._traj_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory', callback_group=self._cb)
        self.get_logger().info('Waiting for trajectory controller ...')
        if not self._traj_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('Timed out waiting for trajectory controller')

        self.get_logger().info('System Ready. Click any pixel to command exact joint positions.')

    def _color_cb(self, msg):
        with self._lock: self._color_img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        with self._lock: self._depth_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _js_cb(self, msg):
        with self._lock: self._latest_js = msg

    def on_click(self, u: int, v: int):
        if self._busy:
            self.get_logger().warn('Arm is moving. Ignored.')
            return

        with self._lock:
            depth_img = self._depth_img
            js        = self._latest_js

        if depth_img is None or js is None:
            return

        raw = depth_img[v, u]
        if raw == 0: return

        # Depth to 3D Math
        z = float(raw) * 0.001 - 0.23
        x = (u - CX) * z / FX + 0.15  
        y = (v - CY) * z / FY + 0.1

        from geometry_msgs.msg import PointStamped
        p_cam = PointStamped()
        p_cam.header.frame_id = CAM_FRAME
        p_cam.point.x, p_cam.point.y, p_cam.point.z = x, y, z

        self.get_logger().info(f"\n--- NEW CLICK REGISTERED ---")
        self.get_logger().info(f"[DEBUG] Raw Pixel: (u={u}, v={v}), Depth z_raw={float(raw)*0.001:.3f}m")
        self.get_logger().info(f"[DEBUG] 1. Point in Camera Frame: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

        try:
            tf_cam = self._tf_buf.lookup_transform(BASE_FRAME, CAM_FRAME, Time(), timeout=Duration(seconds=1.0))
            tf_ee = self._tf_buf.lookup_transform(BASE_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return

        # Log current EE position for reference
        curr_ee_pos = tf_ee.transform.translation
        self.get_logger().info(f"[DEBUG] 2. Current EE Position (Base Frame): X={curr_ee_pos.x:.3f}, Y={curr_ee_pos.y:.3f}, Z={curr_ee_pos.z:.3f}")

        # Calculate pure Cartesian target
        p_base = do_transform_point(p_cam, tf_cam)
        p_target = np.array([p_base.point.x, p_base.point.y, p_base.point.z])
        
        self.get_logger().info(f"[DEBUG] 3. Click Target (Base Frame): X={p_target[0]:.3f}, Y={p_target[1]:.3f}, Z={p_target[2]:.3f}")

        r = tf_ee.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        tcp_target = p_target - R @ POINTER_OFFSET
        
        if np.any(POINTER_OFFSET):
            self.get_logger().info(f"[DEBUG] 4. Final TCP Target (with offset): X={tcp_target[0]:.3f}, Y={tcp_target[1]:.3f}, Z={tcp_target[2]:.3f}")

        self._publish_marker(tcp_target)

        # Dispatch execution to a background thread
        threading.Thread(target=self._solve_ik_and_move, args=(tcp_target, js), daemon=True).start()

    def _publish_marker(self, pos: np.ndarray, delete=False):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'click_target', 0
        m.action = Marker.DELETE if delete else Marker.ADD
        if not delete:
            m.type = Marker.SPHERE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color = ColorRGBA(r=1.0, g=0.3, b=0.0, a=0.9)
        self._marker_pub.publish(m)

    def _solve_ik_and_move(self, tcp_target: np.ndarray, current_js: JointState):
        self._busy = True
        try:
            print(f'\n>>> Preparing IK Request for Target XYZ: ({tcp_target[0]:+.3f}, {tcp_target[1]:+.3f}, {tcp_target[2]:+.3f}) m')
            
            # Log current joints
            cur_map = dict(zip(current_js.name, current_js.position))
            cur_rads = [cur_map.get(n, 0.0) for n in ARM_JOINT_NAMES]
            self.get_logger().info(f"[DEBUG] Current Joint States (rads): {[f'{r:.3f}' for r in cur_rads]}")
            
            # Step 1: Request Inverse Kinematics
            ik_req = PositionIKRequest()
            ik_req.group_name = MOVEIT_GROUP
            ik_req.robot_state.joint_state = current_js
            ik_req.avoid_collisions = True
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = BASE_FRAME
            pose_stamped.pose.position.x = float(tcp_target[0])
            pose_stamped.pose.position.y = float(tcp_target[1])
            pose_stamped.pose.position.z = float(tcp_target[2])
            pose_stamped.pose.orientation.w = 1.0 # Assuming default orientation, you may want to copy current EE orientation
            
            ik_req.pose_stamped = pose_stamped
            ik_req.timeout = DurationMsg(sec=1, nanosec=0)

            req = GetPositionIK.Request()
            req.ik_request = ik_req

            self.get_logger().info("[DEBUG] Request sent to /compute_ik. Waiting for solver...")
            future = self._ik_client.call_async(req)
            
            # Wait for IK result
            while not future.done():
                time.sleep(0.01)
                
            res = future.result()
            if res.error_code.val != 1:
                self.get_logger().error(f"[ERROR] IK Failed (Error Code: {res.error_code.val}). Point is likely unreachable or in collision.")
                return

            # Extract exact joint positions (in radians) from the IK solution
            ik_solution = res.solution.joint_state
            goal_map = dict(zip(ik_solution.name, ik_solution.position))
            target_radians = [goal_map[n] for n in ARM_JOINT_NAMES if n in goal_map]

            self.get_logger().info(f"[DEBUG] IK Solver Success! Target Joint Radians: {[f'{r:.3f}' for r in target_radians]}")
            
            ans = input('>>> Execute joint position command? [y/N]: ').strip().lower()
            if ans != 'y':
                print('>>> Skipped.')
                return

            # Step 2: Send direct Joint Position Command
            self._send_position_command(target_radians)

        except Exception as exc:
            self.get_logger().error(f'Exception: {exc}')
        finally:
            # self._publish_marker(np.zeros(3), delete=True)
            self._busy = False

    def _send_position_command(self, target_radians: list):
        """Sends a single-point position command to the joints."""
        pt = JointTrajectoryPoint()
        pt.positions = target_radians
        pt.time_from_start = DurationMsg(sec=int(MOVE_DUR_S), nanosec=0)

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ARM_JOINT_NAMES
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info("[DEBUG] Dispatching position radians to /arm_controller/follow_joint_trajectory...")
        f = self._traj_client.send_goal_async(goal)
        
        while not f.done():
            time.sleep(0.01)
            
        gh = f.result()
        if not gh.accepted:
            self.get_logger().error('[ERROR] Position command rejected by arm controller.')
            return

        res_f = gh.get_result_async()
        while not res_f.done():
            time.sleep(0.01)

        rw = res_f.result()
        if rw and rw.result.error_code == 0:
            self.get_logger().info('>>> Move Complete: Joint position reached successfully.')
        else:
            self.get_logger().warn(f'>>> Move Finished with error_code={rw.result.error_code}')

    def get_frame(self):
        with self._lock: return self._color_img

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
            label = 'SOLVING / MOVING...' if node._busy else 'Click to point'
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