#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory
from kinova_msgs.msg import JointVelocity


class KinovaTrajectoryBridge(Node):
    def __init__(self):
        super().__init__('kinova_trajectory_bridge')

        self._joint_names = [
            'j2s6s200_joint_1',
            'j2s6s200_joint_2',
            'j2s6s200_joint_3',
            'j2s6s200_joint_4',
            'j2s6s200_joint_5',
            'j2s6s200_joint_6',
        ]

        self._vel_pub = self.create_publisher(
            JointVelocity,
            '/j2s6s200_driver/in/joint_velocity',
            10
        )

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            self._execute_cb
        )

        self.get_logger().info('Kinova trajectory bridge ready.')

    def _publish_velocity(self, velocities):
        msg = JointVelocity()
        msg.joint1 = float(velocities[0])
        msg.joint2 = float(velocities[1])
        msg.joint3 = float(velocities[2])
        msg.joint4 = float(velocities[3])
        msg.joint5 = float(velocities[4])
        msg.joint6 = float(velocities[5])
        self._vel_pub.publish(msg)

    def _execute_cb(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        points = trajectory.points
        joint_order = trajectory.joint_names

        # Map trajectory joint order → driver joint order
        idx_map = [joint_order.index(j) for j in self._joint_names]

        self.get_logger().info(f'Executing trajectory with {len(points)} waypoints.')

        start_time = self.get_clock().now()

        for i, point in enumerate(points):
            t_target = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9

            # Wait until this waypoint's scheduled time
            while True:
                elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
                if elapsed >= t_target:
                    break
                rclpy.spin_once(self, timeout_sec=0.001)

            # Use provided velocities if available, else finite difference
            if point.velocities and len(point.velocities) == len(self._joint_names):
                velocities = [point.velocities[idx_map[j]] for j in range(len(self._joint_names))]
            elif i + 1 < len(points):
                next_point = points[i + 1]
                dt = (
                    next_point.time_from_start.sec + next_point.time_from_start.nanosec * 1e-9
                    - t_target
                )
                if dt > 1e-6:
                    velocities = [
                        (next_point.positions[idx_map[j]] - point.positions[idx_map[j]]) / dt
                        for j in range(len(self._joint_names))
                    ]
                else:
                    velocities = [0.0] * len(self._joint_names)
            else:
                velocities = [0.0] * len(self._joint_names)

            self._publish_velocity(velocities)

        # Stop
        self._publish_velocity([0.0] * 6)
        self.get_logger().info('Trajectory execution complete.')

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result


def main():
    rclpy.init()
    node = KinovaTrajectoryBridge()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()