#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

FEEDING_RAD = [
      2.69807965934403
, 4.241848895816258
, 1.2520215369507008
, 1.2014285336160841
, 4.778086501315855
, 5.36437821950349
]


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        ok = node.move_with_orientation_lock(FEEDING_RAD, DUR_TO_FEEDING, 'TO_FEEDING')
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
