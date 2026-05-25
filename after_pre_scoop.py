#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

FEEDING_RAD = [
      2.2821628729458863
, 4.006647030455947
, 1.1597781589058445
, -4.418303291742182
, 3.85273975526696
, 5.37777072399328
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
