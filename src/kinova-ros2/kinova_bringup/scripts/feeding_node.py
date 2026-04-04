#!/usr/bin/env python3
"""
Feeding Node — SRHT global planner
Trajectory topic: /joint_trajectory_controller/joint_trajectory
  (from ros2_controllers.yaml → joint_trajectory_controller)
"""

import rclpy
from rclpy.node import Node
import numpy as np
import os

from sensor_msgs.msg     import JointState
from geometry_msgs.msg   import PointStamped
from std_msgs.msg        import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from srht_planner import SRHTPlanner

URDF_PATH = '/tmp/j2n6s300.urdf'

# From ros2_controllers.yaml → joint_trajectory_controller → joints
JOINT_NAMES = [
    'j2n6s300_joint_1',
    'j2n6s300_joint_2',
    'j2n6s300_joint_3',
    'j2n6s300_joint_4',
    'j2n6s300_joint_5',
    'j2n6s300_joint_6',
]

# From ros2_controllers.yaml → joint_trajectory_controller topic
TRAJ_TOPIC = '/joint_trajectory_controller/joint_trajectory'


class FeedingNode(Node):
    def __init__(self):
        super().__init__('feeding_node')

        if not os.path.exists(URDF_PATH):
            self.get_logger().error(
                f"URDF missing: {URDF_PATH}\n"
                "Run: xacro ~/kinova_ws/src/kinova-ros2/kinova_description/"
                "urdf/j2n6s300_standalone.xacro > /tmp/j2n6s300.urdf")
            raise FileNotFoundError(URDF_PATH)

        self.planner   = SRHTPlanner(URDF_PATH)
        self.q_current = np.zeros(6)
        self.head_pos  = np.array([0.4, 0.0, 0.6])
        self.executing = False

        # Subscribers
        # Joint state from joint_state_broadcaster
        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_cb, 10)

        # Head position from overhead RGB-D camera
        self.create_subscription(
            PointStamped,
            '/head_pose',
            self._head_cb, 10)

        # Planning goal
        self.create_subscription(
            JointState,
            '/srht/goal_joint',
            self._goal_cb, 10)

        # Publishers
        self.traj_pub   = self.create_publisher(
            JointTrajectory, TRAJ_TOPIC, 10)
        self.status_pub = self.create_publisher(
            String, '/srht/status', 10)

        self.get_logger().info(
            f"FeedingNode ready\n"
            f"  Trajectory → {TRAJ_TOPIC}\n"
            f"  Send goal:\n"
            f"    ros2 topic pub /srht/goal_joint sensor_msgs/JointState "
            f"'{{name: [j1], position: [0.5, 0.3, -0.3, 0.8, 0.2, 0.1]}}' --once")

    def _joint_cb(self, msg):
        # joint_state_broadcaster publishes all joints; extract arm joints by name
        pos_map = dict(zip(msg.name, msg.position))
        for i, jname in enumerate(JOINT_NAMES):
            if jname in pos_map:
                self.q_current[i] = pos_map[jname]

    def _head_cb(self, msg):
        self.head_pos = np.array([msg.point.x, msg.point.y, msg.point.z])

    def _goal_cb(self, msg):
        if self.executing:
            self.get_logger().warn("Already executing — ignoring goal")
            return
        if len(msg.position) < 6:
            self.get_logger().error("Goal needs 6 joint positions")
            return
        self.executing = True
        self._plan_and_execute(np.array(msg.position[:6]))
        self.executing = False

    def _plan_and_execute(self, q_goal):
        self._status("PLANNING")
        path = self.planner.plan(
            q_start=self.q_current,
            q_goal=q_goal,
            obstacle_positions=[self.head_pos],
            timeout=10.0)

        if path is None:
            self.get_logger().error("SRHT: no path found")
            self._status("FAILED")
            return

        self._status("EXECUTING")
        self._publish_traj(path)
        self._status("DONE")

    def _publish_traj(self, path, dt=0.1):
        traj             = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        for i, q in enumerate(path):
            pt              = JointTrajectoryPoint()
            pt.positions    = [float(v) for v in q]
            t               = i * dt
            pt.time_from_start = Duration(
                sec=int(t), nanosec=int((t % 1) * 1e9))
            traj.points.append(pt)
        self.traj_pub.publish(traj)
        self.get_logger().info(
            f"Published {len(path)} waypoints to {TRAJ_TOPIC}")

    def _status(self, s):
        m = String(); m.data = s
        self.status_pub.publish(m)
        self.get_logger().info(f"Status: {s}")


def main(args=None):
    rclpy.init(args=args)
    node = FeedingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
