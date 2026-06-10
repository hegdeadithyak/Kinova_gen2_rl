#!/usr/bin/env python3
import sys
import rclpy
from robot import FeedingOrchestrator, DUR_TO_FEEDING

PICK_RAD = [
      2.877301345958657
, 3.828006445591176
, 1.74634511479214
, -9.59300878156525
, 2.1700887984759727
, 5.23718724373796
]


def main():
    rclpy.init()
    node = FeedingOrchestrator()

    try:
        ok = node.move_with_orientation_lock(PICK_RAD, DUR_TO_FEEDING, 'TO_PICK')
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        sys.exit(130)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
