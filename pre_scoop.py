#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

FEEDING_RAD = [
      2.5744562541747698
, 3.974333032737253
, 1.1287371521882077
, 1.6905496262680537
, 3.928425994498685
, 5.489848793204829
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
