#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateRelay(Node):
    def __init__(self):
        super().__init__('joint_state_relay')
        self._pub = self.create_publisher(JointState, '/joint_states', 10)
        self._sub = self.create_subscription(
            JointState,
            '/j2s6s200_driver/out/joint_state',
            self._cb,
            10
        )
        self.get_logger().info('Joint state relay ready.')

    def _cb(self, msg):
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = JointStateRelay()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()