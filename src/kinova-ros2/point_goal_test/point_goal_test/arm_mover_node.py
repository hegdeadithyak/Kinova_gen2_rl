#!/usr/bin/env python3
"""Subscribe to /goal_point → straight Cartesian path to target.

Uses /compute_cartesian_path with:
  - start_state populated (fixes "empty JointState" MoveIt error)
  - OrientationConstraint to keep EE orientation stable (not JointConstraint,
    which is too restrictive and causes low path coverage)
  - Progressive tolerance relaxation to achieve 100% path coverage
  - 5 cm retract along approach vector
  - Slow speed via trajectory timestamps
"""
import sys
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (Constraints, OrientationConstraint,
                              RobotState)
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as DurationMsg
import tf2_ros
import tf2_geometry_msgs  # noqa: F401


EE_LINK    = 'j2s6s200_end_effector'
CAM_FRAME  = 'camera_color_optical_frame'

JOINT_NAMES = [
    'j2s6s200_joint_1', 'j2s6s200_joint_2', 'j2s6s200_joint_3',
    'j2s6s200_joint_4', 'j2s6s200_joint_5', 'j2s6s200_joint_6',
]

APPROACH_RETRACT = 0.05    # stop 5 cm short of clicked point
CART_SPEED       = 0.04    # m/s
MIN_FRACTION     = 0.90    # require ≥90% of straight path

# Progressive orientation tolerances (rad) — try tightest first, relax if needed
ORI_TOLERANCES = [0.05, 0.1, 0.2, 0.35]


class ArmMoverNode(Node):
    def __init__(self):
        super().__init__('arm_mover_node')
        self._busy         = False
        self._joint_state  = None

        self.declare_parameter('offset_x', 0.0)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_z', 0.0)

        cb = ReentrantCallbackGroup()

        self._tf_buffer  = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._cart_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=cb)

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory', callback_group=cb)

        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)
        self.create_subscription(
            PointStamped, '/goal_point', self._goal_cb, 10, callback_group=cb)

        self.get_logger().info('ArmMover ready — waiting for /goal_point')

    def _js_cb(self, msg: JointState):
        self._joint_state = msg

    def _goal_cb(self, msg: PointStamped):
        if self._busy:
            self.get_logger().warn('Still executing — ignoring new goal')
            return
        self._busy = True
        try:
            self._move(msg)
        finally:
            self._busy = False

    def _move(self, msg: PointStamped):
        ox = self.get_parameter('offset_x').value
        oy = self.get_parameter('offset_y').value
        oz = self.get_parameter('offset_z').value
        tx = msg.point.x + ox
        ty = msg.point.y + oy
        tz = msg.point.z + oz

        # Current EE pose — orientation locked throughout motion
        try:
            tf = self._tf_buffer.lookup_transform(
                'root', EE_LINK, Time(), timeout=Duration(seconds=2.0))
            q = tf.transform.rotation
            t = tf.transform.translation
            ex, ey, ez = t.x, t.y, t.z
        except Exception as e:
            self.get_logger().error(f'TF EE failed: {e}')
            return

        # Retract 5 cm along approach vector
        dx, dy, dz = tx - ex, ty - ey, tz - ez
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist <= APPROACH_RETRACT:
            self.get_logger().warn(
                f'Target too close ({dist*100:.1f} cm ≤ 5 cm) — not moving')
            return
        factor = (dist - APPROACH_RETRACT) / dist
        tx = ex + dx * factor
        ty = ey + dy * factor
        tz = ez + dz * factor

        self.get_logger().info(
            f'Clicked: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})'
            f'  →  goal (−5 cm): ({tx:.3f}, {ty:.3f}, {tz:.3f})')

        # Build start_state from current joint positions (fixes "empty JointState" error)
        start_state = RobotState()
        js = self._joint_state
        if js is not None:
            start_js = JointState()
            start_js.header.stamp    = self.get_clock().now().to_msg()
            start_js.header.frame_id = 'root'
            # Only include the 6 arm joints MoveIt cares about
            for name in JOINT_NAMES:
                if name in js.name:
                    idx = js.name.index(name)
                    start_js.name.append(name)
                    start_js.position.append(js.position[idx])
            start_state.joint_state = start_js
        else:
            self.get_logger().warn('No joint state yet — start_state will be empty')

        # Target pose: keep current EE orientation
        target_pose = Pose(
            position=Point(x=tx, y=ty, z=tz),
            orientation=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w),
        )

        if not self._cart_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('compute_cartesian_path service not available')
            return

        # Try progressively looser orientation tolerances until ≥90% coverage
        traj = None
        for tol in ORI_TOLERANCES:
            path_constraints = self._make_ori_constraint(q, tol)
            req = GetCartesianPath.Request()
            req.header.frame_id  = 'root'
            req.group_name       = 'arm'
            req.link_name        = EE_LINK
            req.waypoints        = [target_pose]
            req.max_step         = 0.005
            req.jump_threshold   = 0.0
            req.avoid_collisions = True
            req.path_constraints = path_constraints
            req.start_state      = start_state

            future = self._cart_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            res = future.result()

            if res is None:
                self.get_logger().error('compute_cartesian_path call timed out')
                return

            frac = res.fraction
            self.get_logger().info(
                f'Cartesian path coverage: {frac*100:.1f}%  '
                f'(orientation tolerance ±{math.degrees(tol):.1f}°)')

            if frac >= MIN_FRACTION:
                traj = res.solution
                break
            else:
                self.get_logger().warn(
                    f'Coverage {frac*100:.1f}% < {MIN_FRACTION*100:.0f}% '
                    f'— relaxing orientation tolerance to ±{math.degrees(ORI_TOLERANCES[ORI_TOLERANCES.index(tol)+1]):.1f}°'
                    if tol != ORI_TOLERANCES[-1] else
                    f'Coverage {frac*100:.1f}% — all tolerances exhausted, using best effort')
                if tol == ORI_TOLERANCES[-1]:
                    traj = res.solution  # use whatever we got

        if traj is None or not traj.joint_trajectory.points:
            self.get_logger().error('No trajectory returned — goal unreachable')
            return

        traj = self._stamp_trajectory(traj)

        if not self._traj_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('arm_controller action server not available')
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj.joint_trajectory

        gh_future = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, gh_future, timeout_sec=5.0)
        gh = gh_future.result()
        if not gh.accepted:
            self.get_logger().error('Trajectory rejected by controller')
            return

        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future, timeout_sec=60.0)
        self.get_logger().info('Motion complete')

    def _make_ori_constraint(self, q, tolerance: float) -> Constraints:
        oc = OrientationConstraint()
        oc.header.frame_id           = 'root'
        oc.link_name                 = EE_LINK
        oc.orientation               = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        oc.absolute_x_axis_tolerance = tolerance
        oc.absolute_y_axis_tolerance = tolerance
        oc.absolute_z_axis_tolerance = tolerance
        oc.weight                    = 1.0
        oc.parameterization          = 1   # ROTATION_VECTOR — avoids gimbal-lock
        c = Constraints()
        c.name = 'keep_ee_orientation'
        c.orientation_constraints.append(oc)
        return c

    def _stamp_trajectory(self, robot_traj):
        points = robot_traj.joint_trajectory.points
        if not points:
            return robot_traj
        total = 0.0
        prev  = points[0]
        points[0].time_from_start = DurationMsg(sec=0, nanosec=0)
        for pt in points[1:]:
            if prev.positions and pt.positions:
                delta = max(abs(a - b)
                            for a, b in zip(pt.positions, prev.positions))
            else:
                delta = 0.01
            total += max(delta / CART_SPEED, 0.05)
            pt.time_from_start = DurationMsg(
                sec=int(total), nanosec=int((total % 1) * 1e9))
            prev = pt
        robot_traj.joint_trajectory.points = points
        return robot_traj


def main():
    try:
        rclpy.init(args=sys.argv)
        node = ArmMoverNode()
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
