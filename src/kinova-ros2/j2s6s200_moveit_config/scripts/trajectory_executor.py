#!/usr/bin/env python3

import time
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

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

        # Shared velocity command (deg/s)
        self._cmd_lock = threading.Lock()
        self._current_cmd = [0.0] * 6
        self._publishing = False

        self._vel_pub = self.create_publisher(
            JointVelocity,
            '/j2s6s200_driver/in/joint_velocity',
            10
        )

        # 100 Hz publisher
        self._timer = self.create_timer(0.01, self._timer_cb)

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            self._execute_cb
        )

        self.get_logger().info('Kinova trajectory bridge ready.')

    # ---------------- TIMER ----------------
    def _timer_cb(self):
        if not self._publishing:
            return

        with self._cmd_lock:
            v = list(self._current_cmd)

        msg = JointVelocity()
        msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6 = v
        msg.joint7 = 0.0

        self._vel_pub.publish(msg)

    # ---------------- UTIL ----------------
    def _set_cmd(self, velocities_rad):
        with self._cmd_lock:
            self._current_cmd = [v * 180.0 / math.pi for v in velocities_rad]

    def _stop(self):
        self._set_cmd([0.0] * 6)
        time.sleep(0.05)
        self._publishing = False

    # ---------------- ACTION ----------------
    def _execute_cb(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        points = trajectory.points
        joint_order = trajectory.joint_names

        idx_map = [joint_order.index(j) for j in self._joint_names]

        self.get_logger().info(f'Executing trajectory with {len(points)} waypoints.')

        self._publishing = True
        start_time = time.monotonic()

        for i, point in enumerate(points):
            t_target = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9

            # Wait until scheduled time
            elapsed = time.monotonic() - start_time
            sleep_dur = t_target - elapsed
            if sleep_dur > 0:
                time.sleep(sleep_dur)

            # Compute velocities
            if point.velocities and len(point.velocities) >= len(self._joint_names):
                velocities = [point.velocities[idx_map[j]] for j in range(6)]
            elif i + 1 < len(points):
                next_pt = points[i + 1]
                dt = (
                    next_pt.time_from_start.sec +
                    next_pt.time_from_start.nanosec * 1e-9 - t_target
                )
                if dt > 1e-6:
                    velocities = [
                        (next_pt.positions[idx_map[j]] - point.positions[idx_map[j]]) / dt
                        for j in range(6)
                    ]
                else:
                    velocities = [0.0] * 6
            else:
                velocities = [0.0] * 6

            self._set_cmd(velocities)

            # ✅ Safe logging (NO shared-state bugs)
            if i < 5 or i == len(points) - 1:
                self.get_logger().info(
                    f'wp {i}/{len(points)} '
                    f't={t_target:.3f}s '
                    f'vel_deg={[round(v * 180 / math.pi, 2) for v in velocities]}'
                )

        self._stop()

        self.get_logger().info('Trajectory execution complete.')

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result


def main():
    rclpy.init()
    node = KinovaTrajectoryBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()