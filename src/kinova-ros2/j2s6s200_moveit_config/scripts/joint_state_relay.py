#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time


class JointStateInterpolator(Node):
    def __init__(self):
        super().__init__('joint_state_interpolator')

        self._last_msg = None
        self._last_time = None

        self._sub = self.create_subscription(
            JointState,
            '/j2s6s200_driver/out/joint_state',
            self._callback,
            10
        )

        self._pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # 50 Hz output
        self._timer = self.create_timer(0.01, self._publish)

        self.get_logger().info('Joint state interpolator running (50 Hz).')

    def _callback(self, msg):
        self._last_msg = msg
        self._last_time = time.time()

    def _publish(self):
        if self._last_msg is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._last_msg.name
        msg.position = self._last_msg.position
        msg.velocity = self._last_msg.velocity
        msg.effort = self._last_msg.effort

        self._pub.publish(msg)


def main():
    rclpy.init()
    node = JointStateInterpolator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()