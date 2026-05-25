#!/usr/bin/env python3
"""Subscribe to /goal_point (root frame) → move arm using phased joint control.

Mirrors click_pointer.py exactly:
  Phase 1 — Y error in camera frame → J2 delta  (elbow pitch)
  Phase 2 — X error in camera frame → J3 delta  (forearm rotation)
  Phase 3 — Z error in camera frame → J1+J4 delta (depth approach)

J5 and J6 are never touched → wrist orientation stays locked.
No MoveIt planning — direct joint trajectory to the controller.
Stops RETRACT_M short of the target along the camera Z axis.
"""
import sys
import threading
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg
import tf2_ros
import tf2_geometry_msgs  # noqa: F401


EE_LINK    = 'j2s6s200_end_effector'
CAM_FRAME  = 'camera_color_optical_frame'

ARM_JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

# Tuned from click_pointer.py
STEP_RADS    = 0.05
CART_DELTA_X = 0.08
CART_DELTA_Y = 0.07
CART_DELTA_Z = 0.06
TICK_DUR_S   = 0.6
ERR_TOL_M    = 0.05
RETRACT_M    = 0.05   # stop 5 cm short along camera Z


class ArmMoverNode(Node):
    def __init__(self):
        super().__init__('arm_mover_node')
        self._busy  = False
        self._joints = {n: 0.0 for n in ARM_JOINT_NAMES}
        self._lock   = threading.Lock()

        self.declare_parameter('offset_x', 0.0)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_z', 0.0)

        cb = ReentrantCallbackGroup()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory', callback_group=cb)

        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10, callback_group=cb)
        self.create_subscription(
            PointStamped, '/goal_point', self._goal_cb, 10, callback_group=cb)

        self.get_logger().info('ArmMover ready — waiting for /goal_point')

    def _js_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._joints:
                    self._joints[name] = pos

    def _goal_cb(self, msg: PointStamped):
        if self._busy:
            self.get_logger().warn('Still executing — ignoring new goal')
            return
        threading.Thread(target=self._move, args=(msg,), daemon=True).start()

    def _move(self, msg: PointStamped):
        self._busy = True
        try:
            ox = self.get_parameter('offset_x').value
            oy = self.get_parameter('offset_y').value
            oz = self.get_parameter('offset_z').value

            # Apply offset then transform target to camera frame
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = msg.point.x + ox
            pt.point.y = msg.point.y + oy
            pt.point.z = msg.point.z + oz

            try:
                target_in_cam = self._tf_buffer.transform(
                    pt, CAM_FRAME, timeout=Duration(seconds=1.0))
                target_cam = [target_in_cam.point.x,
                              target_in_cam.point.y,
                              target_in_cam.point.z]
            except Exception as e:
                self.get_logger().error(f'TF root→camera failed: {e}')
                return

            # EE position in camera frame (= where the camera origin is)
            try:
                tf_ee = self._tf_buffer.lookup_transform(
                    CAM_FRAME, EE_LINK, Time(), timeout=Duration(seconds=1.0))
                start_cam = [tf_ee.transform.translation.x,
                             tf_ee.transform.translation.y,
                             tf_ee.transform.translation.z]
            except Exception as e:
                self.get_logger().error(f'TF EE→camera failed: {e}')
                return

            self.get_logger().info(
                f'Target (cam): x={target_cam[0]:+.3f}  y={target_cam[1]:+.3f}'
                f'  z={target_cam[2]:+.3f}')

            self._perform_phased(start_cam, target_cam)
            self.get_logger().info('Motion complete')

        except Exception as e:
            self.get_logger().error(f'_move failed: {e}')
            import traceback; traceback.print_exc()
        finally:
            self._busy = False

    def _perform_phased(self, start, endpoint):
        MIN_DUR = 1.5
        dy = endpoint[1] - start[1]
        dz = (endpoint[2] - start[2]) - RETRACT_M   # 5 cm retract along depth
        dx_cam = endpoint[0]                          # lateral offset from cam centre

        # Phase 1 — Y (up/down) → J2
        self.get_logger().info(f'Phase 1: Y-error={dy:+.3f} m → J2')
        if abs(dy) > ERR_TOL_M:
            delta_J2 = -(dy / CART_DELTA_Y) * STEP_RADS
            dur = max(MIN_DUR, abs(dy) / CART_DELTA_Y * TICK_DUR_S)
            self._dispatch({1: delta_J2}, dur)

        # Phase 2 — X (left/right) → J3
        self.get_logger().info(f'Phase 2: X-cam={dx_cam:+.3f} m → J3')
        if abs(dx_cam) > ERR_TOL_M:
            delta_J3 = -(dx_cam / CART_DELTA_X) * STEP_RADS
            dur = max(MIN_DUR, abs(dx_cam) / CART_DELTA_X * TICK_DUR_S)
            self._dispatch({2: delta_J3}, dur)

        # Phase 3 — Z (depth approach, minus 5 cm) → J1 + J4
        self.get_logger().info(f'Phase 3: Z-depth={dz:+.3f} m → J1+J4')
        if dz > ERR_TOL_M:
            delta = (dz / CART_DELTA_Z) * STEP_RADS * 2.0
            dur = max(MIN_DUR, dz / CART_DELTA_Z * TICK_DUR_S)
            self._dispatch({0: delta, 3: delta}, dur)

    def _dispatch(self, deltas: dict, duration_s: float):
        with self._lock:
            current = [self._joints[n] for n in ARM_JOINT_NAMES]
        target = list(current)
        for idx, delta in deltas.items():
            target[idx] += delta

        pt0 = JointTrajectoryPoint()
        pt0.positions = current
        pt0.time_from_start = DurationMsg(sec=0, nanosec=0)

        pt1 = JointTrajectoryPoint()
        pt1.positions = target
        pt1.time_from_start = DurationMsg(
            sec=int(duration_s),
            nanosec=int((duration_s % 1) * 1e9))

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = ARM_JOINT_NAMES
        traj.points       = [pt0, pt1]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        done = threading.Event()

        def _on_goal(future):
            gh = future.result()
            if not gh.accepted:
                done.set()
                return
            gh.get_result_async().add_done_callback(lambda _: done.set())

        self._traj_client.send_goal_async(goal).add_done_callback(_on_goal)
        done.wait()


def main():
    try:
        rclpy.init(args=sys.argv)
        node = ArmMoverNode()
        node.get_logger().info('ArmMoverNode initialized')
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        print(f'Critical failure in arm_mover_node: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
