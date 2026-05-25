#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

FEEDING_RAD = [
      2.5527349796918655
, 4.2245047929010076
, 1.2382259630299268
, -5.091467591004313
, 4.773864325723642
, 5.400098133940288
]


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        ok = node.move_joint_space(FEEDING_RAD, DUR_TO_FEEDING, 'TO_FEEDING')
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
