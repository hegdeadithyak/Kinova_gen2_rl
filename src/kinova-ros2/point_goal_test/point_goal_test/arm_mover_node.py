#!/usr/bin/env python3
"""Subscribe to /goal_point → move arm EE to that XYZ.

Planning priority:
  1. PILZ PTP  – smooth S-curve, all joints start/stop together, deterministic
  2. OMPL + orientation – sampling-based fallback with orientation preserved
  3. OMPL position-only – last resort when goal pose has no IK solution
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from pymoveit2 import MoveIt2


JOINT_NAMES = [
    'j2s6s200_joint_1',
    'j2s6s200_joint_2',
    'j2s6s200_joint_3',
    'j2s6s200_joint_4',
    'j2s6s200_joint_5',
    'j2s6s200_joint_6',
]

PILZ  = 'pilz_industrial_motion_planner'
OMPL  = 'ompl'


class ArmMoverNode(Node):
    def __init__(self):
        super().__init__('arm_mover_node')
        self._busy = False
        self.declare_parameter('offset_x', 0.0)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_z', 0.0)

        cb_group = ReentrantCallbackGroup()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name='root',
            end_effector_name='j2s6s200_end_effector',
            group_name='arm',
            callback_group=cb_group,
        )
        self._moveit2.allowed_planning_time = 5.0
        self._moveit2.max_velocity = 0.5
        self._moveit2.max_acceleration = 0.5

        self.create_subscription(PointStamped, '/goal_point', self._goal_cb, 10,
                                 callback_group=cb_group)
        self.get_logger().info('ArmMover ready — waiting for /goal_point')

    def _goal_cb(self, msg: PointStamped):
        if self._busy:
            self.get_logger().warn('Still executing previous goal — ignoring')
            return
        self._busy = True
        try:
            self._move(msg)
        finally:
            self._busy = False

    def _zero_vel_joint_state(self):
        """Return current joint positions with velocities zeroed — required by PILZ."""
        js = self._moveit2.joint_state
        if js is None:
            return None
        zjs = JointState()
        zjs.name     = list(js.name)
        zjs.position = list(js.position)
        zjs.velocity = [0.0] * len(js.position)
        return zjs

    def _plan(self, pipeline, planner_id, **kwargs):
        self._moveit2.pipeline_id = pipeline
        self._moveit2.planner_id  = planner_id
        if pipeline == PILZ:
            kwargs.setdefault('start_joint_state', self._zero_vel_joint_state())
        return self._moveit2.plan(**kwargs)

    def _move(self, msg: PointStamped):
        ox = self.get_parameter('offset_x').value
        oy = self.get_parameter('offset_y').value
        oz = self.get_parameter('offset_z').value
        x = msg.point.x + ox
        y = msg.point.y + oy
        z = msg.point.z + oz
        self.get_logger().info(
            f'Goal: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})'
            f' + offset → ({x:.3f}, {y:.3f}, {z:.3f})')

        try:
            tf = self._tf_buffer.lookup_transform(
                'root', 'j2s6s200_end_effector',
                Time(), timeout=rclpy.duration.Duration(seconds=2.0))
            q = tf.transform.rotation
            quat = [q.x, q.y, q.z, q.w]
        except Exception as e:
            self.get_logger().error(f'Failed to get EE pose: {e}')
            return

        position = [x, y, z]

        # Tier 1 — PILZ PTP: smooth S-curve, synchronized joints, no randomness
        self.get_logger().info('Trying PILZ PTP...')
        traj = self._plan(PILZ, 'PTP', position=position, quat_xyzw=quat)

        # Tier 2 — OMPL with orientation constraint
        if traj is None:
            self.get_logger().warn('PILZ PTP failed → OMPL with orientation')
            traj = self._plan(OMPL, 'RRTConnectkConfigDefault',
                              position=position, quat_xyzw=quat)

        # Tier 3 — OMPL position-only (no orientation constraint)
        if traj is None:
            self.get_logger().warn('OMPL pose failed → OMPL position-only')
            traj = self._plan(OMPL, 'RRTConnectkConfigDefault', position=position)

        if traj is None:
            self.get_logger().error('All planning attempts failed — goal may be unreachable')
            return

        self._moveit2.execute(traj)
        self._moveit2.wait_until_executed()
        self.get_logger().info('Motion complete')


def main():
    try:
        rclpy.init(args=sys.argv)
        node = ArmMoverNode()
        node.get_logger().info('ArmMoverNode initialized')
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
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
